from elin_df.global_parameter import global_parameters
from elin_df.Elin_Distribution_Function import ElinDistributionFunction

def test_global_parameters_basic():
    p = global_parameters()
    # basic consistency checks
    assert p.N_species >= 1
    assert p.nX > 0

def test_elin_distribution_init():
    # just make sure we can construct the object and a key method exists
    df = ElinDistributionFunction(i_species=0, v_soliton=0.0)
    assert isinstance(df, ElinDistributionFunction)
    # check for a core method used elsewhere in the code
    assert hasattr(df, "produce_TranTrappedDisFun_itsMoments") or hasattr(
        df, "return_DfVxExEn_4arrays"
    )
