import asyncio
import json
import time
from dataclasses import asdict, fields, is_dataclass
from typing import Any

import structlog
from config import ConfigAssistant
from data.assistant_dataclasses import *
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import WriteOptions


class DatabaseManager:
    def __init__(
        self,
        data_event_queue: asyncio.Queue | None,
        opt: ConfigAssistant,
        measurement_event_queue: asyncio.Queue | None = None,
    ):
        self.app_start_timestamp = "1970-01-01T00:00:00Z"
        self.logger = structlog.get_logger()
        self.opt = opt
        self.data_event_queue = data_event_queue
        self.measurement_event_queue = measurement_event_queue
        try:
            self.client = InfluxDBClient(
                url=self.opt.influxdb_url,
                token=self.opt.influxdb_token,
                org=self.opt.influxdb_org,
            )
            self.write_api = self.client.write_api(
                write_options=WriteOptions(
                    batch_size=500,
                    flush_interval=1_000,
                    max_retries=3,
                ),
                success_callback=lambda conf, data:
                    self.logger.info("InfluxDB batch written"),
                error_callback=lambda conf, data, exception:
                    self.logger.error(
                        "InfluxDB batch write failed",
                        error=str(exception),
                        data=data,
                    ),
            )
            self.logger.info("Connected to InfluxDB")
        except Exception as e:
            self.logger.error("Failed to connect to InfluxDB", error=str(e))

        # create a storage of recent points to keep track of what has been written
        self.current_state: dict[str, Any] = {}

    def close(self):
        self.write_api.close()
        self.client.close()

    async def run(self) -> None:
        if self.data_event_queue is None:
            return
        while True:
            event = await self.data_event_queue.get()
            if event.get("type") == "data_received":
                self.logger.debug("Received data event", data=event["data"])
                self.write_dataclass(event["data"])

    @staticmethod
    def _influx_field_value(value: Any) -> str | bool | int | float:
        if isinstance(value, (str, bool, int, float)):
            return value

        return json.dumps(
            value,
            default=lambda item: asdict(item) if is_dataclass(item) else str(item),
            separators=(",", ":"),
        )

    @staticmethod
    def _nested_knowledge_values(data: Any, prefix: str = "") -> dict[str, Any]:
        values: dict[str, Any] = {}
        for data_field in fields(data):
            value = getattr(data, data_field.name)
            if value is None:
                continue
            path = f"{prefix}.{data_field.name}" if prefix else data_field.name
            if data_field.metadata.get("knowledge", False):
                values[path] = value
            elif is_dataclass(value):
                values.update(DatabaseManager._nested_knowledge_values(value, path))
        return values

    def write_dataclass(self, data: Any) -> None:
        if not is_dataclass(data):
            self.logger.warning("Ignoring non-dataclass data event", data=data)
            return

        values = {
            field.name: getattr(data, field.name)
            for field in fields(data)
            if getattr(data, field.name) is not None
        }
        knowledge_values = self._nested_knowledge_values(data)
        if not values:
            self.logger.warning(
                "Ignoring data event without values", data_type=type(data).__name__
            )
            return

        name = type(data).__name__
        point = Point(name).time(time.time_ns(), write_precision=WritePrecision.NS)
        for field_name, field_value in values.items():
            point.field(field_name, self._influx_field_value(field_value))
        for field_name, field_value in knowledge_values.items():
            if field_name not in values:
                point.field(field_name, self._influx_field_value(field_value))
        self.logger.warning("Writing telemetry point", data_type=name, values=values, time=time.time_ns(), write_precision=WritePrecision.NS)
        self.write_point(point)
        self.current_state[name] = data

        if self.measurement_event_queue is not None:
            self.measurement_event_queue.put_nowait(
                {
                    "type": "measurements_updated",
                    "name": name,
                    "values": {**values, **knowledge_values},
                }
            )

    def write_measure(self, name: str, measure: str, value: Any):
        if name not in self.current_state:
            # create an object of type name and store it in the current state
            # name is one of the classes in assistant_dataclasses.py
            try:
                self.current_state[name] = globals()[name]()
            except KeyError:
                self.logger.error(f"Class {name} not found in assistant_dataclasses.py")
                return
        setattr(self.current_state[name], measure, value)
        point = Point(name).time(time.time_ns())
        point.field(measure, self._influx_field_value(value))
        self.logger.info(f"Writing point for {name}.{measure} with value {value}")
        self.write_point(point)

        # Keep compatibility with callers that write one measurement at a time.
        if self.measurement_event_queue is not None:
            self.measurement_event_queue.put_nowait(
                {
                    "type": "measurements_updated",
                    "name": name,
                    "values": {measure: value},
                }
            )

    def write_point(self, point: Point):
        self.write_api.write(
            bucket=self.opt.influxdb_bucket,
            org=self.opt.influxdb_org,
            record=point,
            write_precision=WritePrecision.NS,
        )

    def run_query(self, query: str):
        result = self.client.query_api().query(org=self.opt.influxdb_org, query=query)
        if not result or len(result) == 0:
            return None
        table = result[0]
        if not table.records or len(table.records) == 0:
            return None
        return table.records[0].get_value()

    def run_records(self, query: str) -> list[dict[str, Any]]:
        result = self.client.query_api().query(org=self.opt.influxdb_org, query=query)
        if not result:
            return []
        return [record.values for table in result for record in table.records]

    def mean(self, name: str, measure: str, range: str = "-30s") -> float | None:
        query = f"""
        from(bucket:"{self.opt.influxdb_bucket}") 
        |> range(start: {range}) 
        |> filter(fn: (r) => r._measurement == "{name}" and r._field == "{measure}") |> mean()
        """
        out = self.run_query(query)
        return out

    def max(self, name: str, measure: str, range: str = "-30s") -> float | None:
        query = f"""
        from(bucket:"{self.opt.influxdb_bucket}") 
        |> range(start: {range}) 
        |> filter(fn: (r) => r._measurement == "{name}" and r._field == "{measure}") |> max()
        """
        out = self.run_query(query)
        return out

    def min(self, name: str, measure: str, range: str = "-30s") -> float | None:
        query = f"""
        from(bucket:"{self.opt.influxdb_bucket}") 
        |> range(start: {range}) 
        |> filter(fn: (r) => r._measurement == "{name}" and r._field == "{measure}") |> min()
        """
        out = self.run_query(query)
        return out

    def stddev(self, name: str, measure: str, range: str = "-30s") -> float | None:
        query = f"""
        from(bucket:"{self.opt.influxdb_bucket}") 
        |> range(start: {range}) 
        |> filter(fn: (r) => r._measurement == "{name}" and r._field == "{measure}") |> stddev()
        """
        out = self.run_query(query)
        return out

    def count(self, name: str, measure: str, range: str = "-30s") -> float | None:
        query = f"""
        from(bucket:"{self.opt.influxdb_bucket}") 
        |> range(start: {range}) 
        |> filter(fn: (r) => r._measurement == "{name}" and r._field == "{measure}") |> count()
        """
        out = self.run_query(query)
        return out

    # trend
    def derivative(self, name: str, measure: str, range: str = "-30s") -> float | None:
        query = f"""
        from(bucket:"{self.opt.influxdb_bucket}") 
        |> range(start: {range}) 
        |> filter(fn: (r) => r._measurement == "{name}" and r._field == "{measure}") |> derivative()
        """
        out = self.run_query(query)
        return out

    # mean of the trend
    def mean_derivative(
        self, name: str, measure: str, range: str = "-30s"
    ) -> float | None:
        query = f"""
        from(bucket:"{self.opt.influxdb_bucket}") 
        |> range(start: {range}) 
        |> filter(fn: (r) => r._measurement == "{name}" and r._field == "{measure}") |> derivative() |> mean()
        """
        out = self.run_query(query)
        return out

    # standard deviation of the trend
    def stddev_derivative(
        self, name: str, measure: str, range: str = "-30s"
    ) -> float | None:
        query = f"""
        from(bucket:"{self.opt.influxdb_bucket}") 
        |> range(start: {range}) 
        |> filter(fn: (r) => r._measurement == "{name}" and r._field == "{measure}") |> derivative() |> stddev()
        """
        out = self.run_query(query)
        return out

    def derivative_quantile(
            self,
            name: str,
            measure: str,
            quantile: float = 0.95,
            range: str = "-10m",
    ) -> float | None:
            query = f'''
            import "math"

            from(bucket:"{self.opt.influxdb_bucket}")
                |> range(start: {range})
                |> filter(fn: (r) =>
                    r._measurement == "{name}" and r._field == "{measure}"
                )
                |> derivative(unit: 1s)
                |> map(fn: (r) => ({{ r with _value: math.abs(x: r._value) }}))
                |> quantile(q: {quantile}, method: "estimate_tdigest")
            '''
            return self.run_query(query)

    @staticmethod
    def _flux_literal(value: str | bool) -> str:
        if isinstance(value, bool):
            return str(value).lower()
        escaped_value = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped_value}"'

    def first_value(
        self, name: str, measure: str, range: str = "-30s"
    ) -> str | bool | None:
        query = f"""
        from(bucket:"{self.opt.influxdb_bucket}")
        |> range(start: {range})
        |> filter(fn: (r) => r._measurement == "{name}" and r._field == "{measure}")
        |> first()
        """
        return self.run_query(query)

    def last_value(
        self, name: str, measure: str, range: str = "-30s"
    ) -> str | bool | None:
        query = f"""
        from(bucket:"{self.opt.influxdb_bucket}")
        |> range(start: {range})
        |> filter(fn: (r) => r._measurement == "{name}" and r._field == "{measure}")
        |> last()
        """
        return self.run_query(query)

    def value_counts(
        self, name: str, measure: str, range: str = "-30s"
    ) -> dict[str | bool, int]:
        query = f"""
        from(bucket:"{self.opt.influxdb_bucket}")
        |> range(start: {range})
        |> filter(fn: (r) => r._measurement == "{name}" and r._field == "{measure}")
        |> map(fn: (r) => ({{r with state: r._value}}))
        |> group(columns: ["state"])
        |> count()
        """
        return {
            record["state"]: int(record["_value"])
            for record in self.run_records(query)
            if record.get("state") is not None and record.get("_value") is not None
        }

    def value_transitions(
        self, name: str, measure: str, range: str = "-30s"
    ) -> int | None:
        query = f"""
        from(bucket:"{self.opt.influxdb_bucket}")
        |> range(start: {range})
        |> filter(fn: (r) => r._measurement == "{name}" and r._field == "{measure}")
        |> sort(columns: ["_time"])
        """
        values = [record.get("_value") for record in self.run_records(query)]
        if not values:
            return None
        return sum(previous != current for previous, current in zip(values, values[1:]))

    def current_value_duration(
        self, name: str, measure: str, value: str | bool, range: str = "-30s"
    ) -> int | None:
        value_literal = self._flux_literal(value)
        query = f"""
        from(bucket:"{self.opt.influxdb_bucket}")
        |> range(start: {range})
        |> filter(fn: (r) => r._measurement == "{name}" and r._field == "{measure}")
        |> sort(columns: ["_time"])
        |> stateDuration(fn: (r) => r._value == {value_literal}, unit: 1s)
        |> filter(fn: (r) => r._value == {value_literal})
        |> map(fn: (r) => ({{r with _value: r.stateDuration}}))
        |> last()
        """
        return self.run_query(query)
