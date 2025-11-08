from .global_parameter import global_parameters
from .Elin_Distribution_Function import ElinDistributionFunction

def main():
    """
    Very small smoke test: initialize parameters and one distribution function
    and print a couple of basic fields.
    """
    params = global_parameters()
    print("Number of species:", params.N_species)
    print("Grid points in x:", params.nX)

    df = ElinDistributionFunction(i_species=0, v_soliton=0.0)
    print("Created ElinDistributionFunction with species index 0")
