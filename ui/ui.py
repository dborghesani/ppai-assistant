import tkinter as tk
from tkinter import ttk
import asyncio
from events import CarEvent
from datetime import datetime

class CarDashboard:
    def __init__(self, root: tk.Tk, agent_queue: asyncio.Queue):
        self.root = root
        self.root.title("Automotive AI Dashboard")
        self.root.geometry("800x600") # Approximate based on 9x9
        self.event_queue = agent_queue # Reference to the asyncio.Queue in Agent

        # Styles
        self.style = ttk.Style()
        self.style.configure("Header.TLabel", font=("Helvetica", 12, "bold"))

        # Grid setup (9x9)
        self._setup_grid()

        # 1. Driving Behavior
        self.setup_radio_button_group("Driving Behavior", [
            ("Safe", "safe"),
            ("Aggressive", "aggressive"),
            ("Non-reactive", "non-reactive")
        ], row=0, col=0, rows=2, cols=3)

        # 2. Driver Activity
        self.setup_radio_button_group("Driver Activity", [
            ("Driving", "driving"),
            ("Eating", "eating"),
            ("On the phone", "phone"),
            ("Falling asleep", "sleep")
        ], row=2, col=0, rows=2, cols=3)

        # 3. Mood
        self.setup_radio_button_group("Mood", [
            ("Happy", "happy"),
            ("Sad", "sad"),
            ("Worried", "worried"),
            ("Scared", "scared"),
            ("Normal", "normal")
        ], row=4, col=0, rows=2, cols=3)

        # 4. Driving Goal
        self.setup_radio_button_group("Driving Goal", [
            ("Driving", "driving"),
            ("Parking", "parking"),
            ("Approaching Dest.", "destination")
        ], row=6, col=0, rows=2, cols=3)

        # 5. Driving Conditions
        # Row 0: Traffic
        self.setup_radio_button_group("Traffic", [
            ("Light", "light"),
            ("Heavy", "heavy")
        ], row=0, col=3, rows=1, cols=2)

        # Row 1: Road
        self.setup_radio_button_group("Road Type", [
            ("Highway", "highway"),
            ("Urban", "urban")
        ], row=1, col=3, rows=1, cols=2)

        # Row 2: Time
        self.setup_radio_button_group("Time", [
            ("Day", "day"),
            ("Night", "night")
        ], row=2, col=3, rows=1, cols=2)

        # Row 3: Weather
        self.setup_radio_button_group("Weather", [
            ("Sunny", "sunny"),
            ("Rainy", "rainy"),
            ("Foggy", "foggy"),
            ("Snowy", "snowy")
        ], row=3, col=3, rows=1, cols=3)

        # 6. Risk Assessment
        self.setup_push_button("Area Status", "Risky Area", 1, 5, 1, 1, bg="red", text_color="white", is_risk=True)

        # 7. Occupants (Persons)
        self.setup_slider("Persons Around", 0, 10, 0, row=5, col=3, rows=1, cols=3)

        # 8. Objects Detected
        self.setup_radio_button_group("Detected Objects", [
            ("None", "none"),
            ("Luggage", "luggage"),
            ("Gun", "gun")
        ], row=6, col=3, rows=2, cols=3)

        # 9. Car Status
        # Speed Slider
        self.setup_slider("Speed (km/h)", 0, 200, 0, row=8, col=0, rows=1, cols=3, max_val=200)
        
        # Engine State
        self.setup_radio_button_group("Engine", [
            ("Off", "off"),
            ("On", "on")
        ], row=9, col=0, rows=1, cols=2)
        
        # Parked State
        self.setup_radio_button_group("Parked", [
            ("No", "no"),
            ("Yes", "yes")
        ], row=9, col=2, rows=1, cols=2)

        # Doors
        self.setup_radio_button_group("Doors", [
            ("Locked", "locked"),
            ("Unlocked", "unlocked")
        ], row=10, col=0, rows=1, cols=2)

        # Internal Temp Slider
        self.setup_slider("Internal Temp", 15, 30, 22, row=10, col=2, rows=1, cols=2)

    def _setup_grid(self):
        # 9x9 grid definition
        # I'll create 9x9 frames/labels to visualize the grid if needed, 
        # but simply using grid() with rowspan/colspan is enough for logic.
        for r in range(9):
            self.root.grid_rowconfigure(r, weight=1, minsize=50)
        for c in range(9):
            self.root.grid_columnconfigure(c, weight=1)

    def setup_radio_button_group(self, title, options, row, col, rows, cols):
        frame = ttk.LabelFrame(self.root, text=title)
        frame.grid(row=row, column=col, rowspan=rows, columnspan=cols, sticky="nsew", padx=2, pady=2)
        
        var = tk.StringVar(value=options[0][1])
        
        for i, (text, value) in enumerate(options):
            rb = ttk.Radiobutton(frame, text=text, variable=var, value=value, 
                                 command=lambda v=value: self._on_radio_change(title, v))
            rb.grid(row=i, column=0, sticky="w", padx=5)
            
        # Store variable for access
        setattr(self, f"var_{title.replace(' ', '_').lower()}", var)

    def setup_slider(self, title, start, end, default, row, col, rows, cols, max_val=None):
        frame = ttk.LabelFrame(self.root, text=title)
        frame.grid(row=row, column=col, rowspan=rows, columnspan=cols, sticky="nsew", padx=2, pady=2)
        
        var = tk.IntVar(value=default)
        if max_val is None: max_val = end
        
        scale = ttk.Scale(frame, from_=start, to=end, orient=tk.HORIZONTAL, 
                          command=lambda v: self._on_slider_change(title, var.get()))
        scale.set(default)
        scale.grid(row=0, column=0, columnspan=3, sticky="we", padx=5)
        
        label = ttk.Label(frame, textvariable=var)
        label.grid(row=1, column=0, columnspan=3)

        setattr(self, f"var_{title.replace(' ', '_').lower()}", var)

    def setup_push_button(self, title, text, row, col, rows, cols, bg="red", text_color="white", is_risk=False):
        frame = ttk.LabelFrame(self.root, text=title)
        frame.grid(row=row, column=col, rowspan=rows, columnspan=cols, sticky="nsew", padx=2, pady=2)
        
        # Simple container to center the button
        container = tk.Frame(frame)
        container.pack(fill=tk.BOTH, expand=True)
        
        btn = tk.Button(container, text=text, bg=bg, fg=text_color, activebackground="#333", activeforeground="white",
                        command=lambda: self._on_button_click(title))
        btn.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def _on_radio_change(self, title, value):
        # Determine event type based on title
        event_type = f"{title.lower().replace(' ', '_')}"
        event = CarEvent(event_type=event_type, value=value)
        self.root.after(0, lambda: asyncio.run_coroutine_threadsafe(self.send_event(event), asyncio.get_event_loop()))

    def _on_slider_change(self, title, value):
        event_type = f"{title.lower().replace(' ', '_')}"
        event = CarEvent(event_type=event_type, value=float(value))
        self.root.after(0, lambda: asyncio.run_coroutine_threadsafe(self.send_event(event), asyncio.get_event_loop()))

    def _on_button_click(self, title):
        event_type = f"{title.lower().replace(' ', '_')}_triggered"
        if title == "Area Status":
            # Special case for button - maybe toggle or just one shot?
            # Prompt said: "push button con l'indicazione se l'area corrente è rischiosa o meno"
            # Let's make it toggle
            pass 
        
        event = CarEvent(event_type=event_type, value=1.0)
        self.root.after(0, lambda: asyncio.run_coroutine_threadsafe(self.send_event(event), asyncio.get_event_loop()))

    async def send_event(self, event):
        await self.event_queue.put(event)