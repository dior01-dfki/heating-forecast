import inspect
from copy import deepcopy

class DefaultParamsMixin:
    @classmethod
    def get_default_params(cls):
            """
            Return default parameters defined in the class's __init__ method signature.
            Ignores *args and **kwargs.
            """
            defaults = {}
            
            # Get the signature of the __init__ method
            sig = inspect.signature(cls.__init__)
            
            for name, parameter in sig.parameters.items():
                # 1. Skip positional-only and variable arguments (*args, **kwargs)
                if parameter.kind in (
                    inspect.Parameter.VAR_POSITIONAL,  # *args
                    inspect.Parameter.VAR_KEYWORD,    # **kwargs
                    inspect.Parameter.POSITIONAL_ONLY
                ):
                    continue

                # 2. Check if the parameter has a default value
                if parameter.default is not inspect.Parameter.empty:
                    defaults[name] = parameter.default
                    
            # We use deepcopy to ensure returned values are not mutable references
            # back to the class's internal state (though typically not an issue here)
            return deepcopy(defaults)
