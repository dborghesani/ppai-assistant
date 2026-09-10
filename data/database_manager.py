from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS
import structlog
from config import ConfigAssistant
import asyncio
from typing import Any
from data.assistant_dataclasses import *

class DatabaseManager:
    def __init__(self, event_queue: asyncio.Queue | None, opt: ConfigAssistant):
        self.app_start_timestamp = "1970-01-01T00:00:00Z"
        self.logger = structlog.get_logger()
        self.opt = opt
        self.event_queue = event_queue
        try:
            self.client = InfluxDBClient(
                url=self.opt.influxdb_url,
                token=self.opt.influxdb_token,
                org=self.opt.influxdb_org
            )
            self.write_api = self.client.write_api(write_options=SYNCHRONOUS)
            self.logger.info("Connected to InfluxDB")
        except Exception as e:
            self.logger.error("Failed to connect to InfluxDB", error=str(e))

        # create a storage of recent points to keep track of what has been written
        self.current_state: dict[str, Any] = {}

    def close(self):
        self.client.close()

    def load_last_state(self, name: str) -> Any:
        # load the last state of the given measurement (class name) from the database
        # add filter with timestamp starting from when the app is started
        query = f"""
        from(bucket: "{self.opt.influxdb_bucket}")
        |> range(start: time(v: "{self.app_start_timestamp}"))
        |> filter(fn: (r) => r._measurement == "{name}")
        |> last()
        """
        tables = self.client.query_api().query(query)
        if len(tables) == 0:
            return None
        try:
            obj = globals()[name]()
        except KeyError:
            self.logger.error(f"Class {name} not found in assistant_dataclasses.py")
            return None
        valid_fields = {field.name for field in obj.__dataclass_fields__.values()}
        # each field of the dataclass comes back in its own table/record, so merge them all
        for table in tables:
            for record in table.records:
                field_name = record.get_field()
                if field_name in valid_fields:
                    setattr(obj, field_name, record.get_value())
        return obj

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
        last_state = self.load_last_state(name)
        if last_state:
            # merge in fields from the last persisted state, without overwriting the value just set
            for field_name in last_state.__dataclass_fields__:
                if field_name != measure:
                    setattr(self.current_state[name], field_name, getattr(last_state, field_name))
        point = Point(self.current_state[name].__class__.__name__)
        for field_name, field_value in self.current_state[name].__dict__.items():
            point.field(field_name, field_value)
        self.logger.info(f"Writing point for {name}.{measure} with value {value}")
        self.write_point(point)

        # emit a signal to inform that a new measurement has been written
        if self.event_queue is not None:
            asyncio.run_coroutine_threadsafe(
                self.event_queue.put({
                    "event": "measure_updated",
                    "name": name,
                    "measure": measure,
                    "value": value,
                }),
                asyncio.get_event_loop(),
            )

    def write_point(self, point: Point):
        self.write_api.write(
            bucket=self.opt.influxdb_bucket,
            org=self.opt.influxdb_org,
            record=point,
        )

    def run_query(self, query: str):
        result = self.client.query_api().query(org=self.opt.influxdb_org, query=query)
        if not result or len(result) == 0:
            return None
        table = result[0]
        if not table.records or len(table.records) == 0:
            return None
        return table.records[0].get_value()

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
    def mean_derivative(self, name: str, measure: str, range: str = "-30s") -> float | None:
        query = f"""
        from(bucket:"{self.opt.influxdb_bucket}") 
        |> range(start: {range}) 
        |> filter(fn: (r) => r._measurement == "{name}" and r._field == "{measure}") |> derivative() |> mean()
        """
        out = self.run_query(query)
        return out

    # standard deviation of the trend
    def stddev_derivative(self, name: str, measure: str, range: str = "-30s") -> float | None:
        query = f"""
        from(bucket:"{self.opt.influxdb_bucket}") 
        |> range(start: {range}) 
        |> filter(fn: (r) => r._measurement == "{name}" and r._field == "{measure}") |> derivative() |> stddev()
        """
        out = self.run_query(query)
        return out



