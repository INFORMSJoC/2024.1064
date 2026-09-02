import json
from dataclasses import asdict, dataclass, is_dataclass

from pwlopt.utils import with_logger


@with_logger
@dataclass(frozen=True, kw_only=True, slots=True)
class HydroData:
    maxT : int # end of time horizon
    nperiod : int # number of periods
    J : list[int] # index of pumps/generators
    deltaT : int # length of timestep (h)
    C : list[int]
    inflow : list[float] # water inflow prediction
    price : list[float] # wholesale electricity price prediction
    pump_cost : list[float] # cost of pumping
    V0 : float # initial water level in reservoir
    VT : float # final water level in reservoir
    Vmin : list[float] # lower bound on water level in reservoir
    Vmax : list[float] # upper bound on water level in reservoir
    U0 : list[float]
    G0 : list[float]
    flow_by_pump : list[float] # amount of water pumped back by pumps
    Qmin : list[float]
    Qmax : list[float]
    ramp_down : list[float]
    ramp_up : list[float]
    max_power_consumed: list[float]
    max_power_produced: list[float]
    Y: list[float]
    W: list[float]
    theta: float
    max_spillage: float

    def to_json(self, include_null=False) -> dict:
        """Converts this to json. Assumes variables are snake cased, converts to camel case.

        Args:
            include_null (bool, optional): Whether null values are included. Defaults to False.

        Returns:
            dict: Json dictionary
        """
        return asdict(
            self,
            dict_factory=lambda fields: {
                key: value
                for (key, value) in fields
                if value is not None or include_null
            },
        )
    
    @classmethod
    def from_file(cls, plantfile:str, pricefile:str, inflowfile:str, pumpcostfile:str):
        """Constructs `this` from given json. Assumes camel case convention is used and converts to camel case.

        Args:
            json (dict): Json dictionary

        Raises:
            ValueError: When `this` isn't a dataclass
        """
        if not is_dataclass(cls):
            raise ValueError(f"{cls.__name__} must be a dataclass")
        
        with open(plantfile, "r") as f:
            cls.logger.info("Read plant file %s", plantfile) # type: ignore[attr-defined]
            plant_data = json.load(f)
            
        with open(pricefile, "r") as f:
            cls.logger.info("Read price file %s", pricefile) # type: ignore[attr-defined]
            price_data = json.load(f)

        with open(inflowfile, "r") as f:
            cls.logger.info("Read inflow file %s", inflowfile) # type: ignore[attr-defined]
            inflow_data = json.load(f)

        with open(pumpcostfile, "r") as f:
            cls.logger.info("Read pump-cost file %s", pumpcostfile) # type: ignore[attr-defined]
            pumpcost_data = json.load(f)
                    
        kwargs = {
            key: value
            for key, value in plant_data.items()
        }

        kwargs["inflow"] = inflow_data
        kwargs["price"] = price_data
        kwargs["pump_cost"] = pumpcost_data

        return cls(**kwargs)
    
    def write_model(self, filename: str) -> None:
        with open(filename, 'w+') as out:
            json.dump(self.to_json(), out, indent=4)

    def print_price_inflow_ampl(self):
        for t in range(self.nperiod):
            print(t+1, self.inflow[t], self.price[t], self.pump_cost[t])