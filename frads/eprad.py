from functools import partial
import json


class Handles:
    def __init__(self):
        self.outdoor_drybulb_temperature = None
        self.direct_normal_irradiance = None
        self.diffuse_horizontal_irradiance = None
        self.complex_fenestration_state = {}
        self.window_actuators = {}


class EnergyPlusSetup:

    def __init__(self, api, epjs):
        self.api = api
        self.epjs = epjs
        self.state = self.api.state_manager.new_state()
        self.handles = Handles()
        self.api.exchange.request_variable(self.state, 'Site Outdoor Air Drybulb Temperature'.encode()) 
        self.api.exchange.request_variable(self.state, 'Site Outdoor Air Drybulb Temperature'.encode())
        self.api.exchange.request_variable(self.state, 'Site Outdoor Air Drybulb Temperature'.encode()) 
        self.api.exchange.callback_begin_new_environment(self.state, self.get_handles())

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.api.state_manager.delete_state(self.state)

    def actuate(self, obj, value):
        self.api.exchange.set_actuator_value(self.state, obj, value)

    def get_variable_value(self, handle):
        return self.api.exchange.get_variable_value(self.state, handle)

    def get_variable_handle(self, name):
        return self.api.exchange.get_variable_handle(self.state, name)

    def get_handles(self):
        def callback_function(state):
            self.handles.outdoor_drybulb_temperature = self.api.exchange.get_variable_handle(state, )
            self.handles.direct_normal_irradiance = self.api.exchange.get_variable_handle(state, )
            self.handles.diffuse_horizontal_irradiance = self.api.exchange.get_variable_handle(state,)
            construction_complex_fenestration_state = self.epjs['Construction:ComplexFenestrationState']
            cfs_handles = {}
            window_actuators = {}
            for cfs in construction_complex_fenestration_state:
                cfs_handles[cfs] = self.api.api.getConstructionHandle(state, cfs.encode())
            window_surfaces = self.epjs['FenestrationSurface:Detailed']
            for window in window_surfaces:
                window_actuators[window] = self.api.exchange.get_actuator_handle(state, window.encode())
            self.handles.complex_fenestration_state = cfs_handles
            self.handles.window_actuators = window_actuators
        return callback_function
        
    def run(self):
        with open("ep.json", "w") as wtr:
            json.dump(self.epjs, wtr)
        self.api.runtime.run_energyplus(self.state, ["-d", "output", "-r", "ep.json"])

    def set_callback(self, method_name: str, func):
        try:
            method = getattr(self.api.runtime, method_name)
        except AttributeError:
            raise AttributeError(f"Method {method_name} not found in EnergyPlus runtime API")
        method(self.state, partial(func, self))
