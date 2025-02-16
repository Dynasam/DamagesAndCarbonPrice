######
# Default parameters
######

from collections import namedtuple


class Params:
    default_params = None

    obj = None

    def __init__(self, **kwargs):

        self.default_params = dict(

            beta = 2.0,
            gamma = 1500.0,
            useCalibratedGamma = True,
            cost_level = 'p50',

            carbonbudget = 0, # GtCO2
            relativeBudget = False,
            budgetYear = 2100,
            noPositiveEmissionsAfterBudgetYear = True,

            carbonbudgetOld = 0,

            progRatio = 0.82,
            exogLearningRate = 0.0,

            # If True, use progRatio to calculate corresponding exog learning rate.
            # The endogenous learning (LBD) is then ignored
            useCalibratedExogLearningRate = False,
            minEmissions = -20, # Default is at most 20 GtCO2/yr net negative emissions
            maxReductParam = 0.05 +100,

            CE_values_num = 800,
            E_values_num = 2,
            E_min_rel = -3, E_max_rel = 2.5,

            K_values_num = 50,
            K_min = 0, K_max = 3000,

            p_values_num = 1000,
            p_values_max_rel = 1.5,

            T = 130,
            start_year = 2020,
            t_values_num = int(130/5)+1,

            T0 = 1.0 + 4 * 0.00062 * 38.8, # Temperature in 2020
            TCRE = 0.62e-3, # degC / GtCO2

            r = 0.015,
            elasmu = 1.001,
            discountConsumptionFixed = False, # Only used when maximise_utility is False

            maximise_utility = True,

            SSP_GDP = 'same', SSP_population='same', SSP_emissions='same',
            SSP = 'SSP2',
            K_start = 223.0,
            useBaselineCO2Intensity = True,

            fastmath = True,

            damage = "nodamage",
            damage_coeff = 0.0, # Only used when damage == 'damageGeneral'

            runname = "default",
            shortname = "default"
        )

        for key, value in kwargs.items():
            if key not in self.default_params:
                raise KeyError("Key " + str(key) +" not a valid argument")
            self.default_params[key] = value

        ## Create namedtuple
        ParamsObj = namedtuple('ParamsObj', sorted(self.default_params))
        self.obj = ParamsObj(**self.default_params)

    def __repr__(self):
        return "Params("+str(self.default_params)+")"

    def dict(self):
        return self.default_params
