#pragma once

#include "integration.hpp"

namespace tools {

template <>
std::vector<std::array<double, 2>> IntegrationRule<1, 0>::rule{
    {0.50000000000000000000, 1.00000000000000000000},
};
template <>
std::vector<std::array<double, 2>> IntegrationRule<1, 1>::rule{
    {0.50000000000000000000, 1.00000000000000000000},
};
template <>
std::vector<std::array<double, 2>> IntegrationRule<1, 2>::rule{
    {0.25000000000000000000, 0.66666666666666662966},
    {0.75000000000000000000, 0.66666666666666662966},
    {0.50000000000000000000, -0.33333333333333331483},
};
template <>
std::vector<std::array<double, 2>> IntegrationRule<1, 3>::rule{
    {0.25000000000000000000, 0.66666666666666662966},
    {0.75000000000000000000, 0.66666666666666662966},
    {0.50000000000000000000, -0.33333333333333331483},
};
template <>
std::vector<std::array<double, 2>> IntegrationRule<1, 4>::rule{
    {0.16666666666666665741, 0.67500000000000004441},
    {0.50000000000000000000, 0.67500000000000004441},
    {0.83333333333333337034, 0.67500000000000004441},
    {0.25000000000000000000, -0.53333333333333332593},
    {0.75000000000000000000, -0.53333333333333332593},
    {0.50000000000000000000, 0.04166666666666666435},
};
template <>
std::vector<std::array<double, 2>> IntegrationRule<1, 5>::rule{
    {0.16666666666666665741, 0.67500000000000004441},
    {0.50000000000000000000, 0.67500000000000004441},
    {0.83333333333333337034, 0.67500000000000004441},
    {0.25000000000000000000, -0.53333333333333332593},
    {0.75000000000000000000, -0.53333333333333332593},
    {0.50000000000000000000, 0.04166666666666666435},
};
template <>
std::vector<std::array<double, 2>> IntegrationRule<1, 6>::rule{
    {0.12500000000000000000, 0.81269841269841269771},
    {0.37500000000000000000, 0.81269841269841269771},
    {0.62500000000000000000, 0.81269841269841269771},
    {0.87500000000000000000, 0.81269841269841269771},
    {0.16666666666666665741, -0.86785714285714288252},
    {0.50000000000000000000, -0.86785714285714288252},
    {0.83333333333333337034, -0.86785714285714288252},
    {0.25000000000000000000, 0.17777777777777778456},
    {0.75000000000000000000, 0.17777777777777778456},
    {0.50000000000000000000, -0.00277777777777777788},
};
template <>
std::vector<std::array<double, 2>> IntegrationRule<1, 7>::rule{
    {0.12500000000000000000, 0.81269841269841269771},
    {0.37500000000000000000, 0.81269841269841269771},
    {0.62500000000000000000, 0.81269841269841269771},
    {0.87500000000000000000, 0.81269841269841269771},
    {0.16666666666666665741, -0.86785714285714288252},
    {0.50000000000000000000, -0.86785714285714288252},
    {0.83333333333333337034, -0.86785714285714288252},
    {0.25000000000000000000, 0.17777777777777778456},
    {0.75000000000000000000, 0.17777777777777778456},
    {0.50000000000000000000, -0.00277777777777777788},
};
template <>
std::vector<std::array<double, 2>> IntegrationRule<1, 8>::rule{
    {0.10000000000000000555, 1.07645778218694876394},
    {0.29999999999999998890, 1.07645778218694876394},
    {0.50000000000000000000, 1.07645778218694876394},
    {0.69999999999999995559, 1.07645778218694876394},
    {0.90000000000000002220, 1.07645778218694876394},
    {0.12500000000000000000, -1.44479717813051156128},
    {0.37500000000000000000, -1.44479717813051156128},
    {0.62500000000000000000, -1.44479717813051156128},
    {0.87500000000000000000, -1.44479717813051156128},
    {0.16666666666666665741, 0.48816964285714287142},
    {0.50000000000000000000, 0.48816964285714287142},
    {0.83333333333333337034, 0.48816964285714287142},
    {0.25000000000000000000, -0.03386243386243386472},
    {0.75000000000000000000, -0.03386243386243386472},
    {0.50000000000000000000, 0.00011574074074074075},
};
template <>
std::vector<std::array<double, 2>> IntegrationRule<1, 9>::rule{
    {0.10000000000000000555, 1.07645778218694876394},
    {0.29999999999999998890, 1.07645778218694876394},
    {0.50000000000000000000, 1.07645778218694876394},
    {0.69999999999999995559, 1.07645778218694876394},
    {0.90000000000000002220, 1.07645778218694876394},
    {0.12500000000000000000, -1.44479717813051156128},
    {0.37500000000000000000, -1.44479717813051156128},
    {0.62500000000000000000, -1.44479717813051156128},
    {0.87500000000000000000, -1.44479717813051156128},
    {0.16666666666666665741, 0.48816964285714287142},
    {0.50000000000000000000, 0.48816964285714287142},
    {0.83333333333333337034, 0.48816964285714287142},
    {0.25000000000000000000, -0.03386243386243386472},
    {0.75000000000000000000, -0.03386243386243386472},
    {0.50000000000000000000, 0.00011574074074074075},
};
template <>
std::vector<std::array<double, 2>> IntegrationRule<1, 10>::rule{
    {0.08333333333333332871, 1.51480519480519482123},
    {0.25000000000000000000, 1.51480519480519482123},
    {0.41666666666666668517, 1.51480519480519482123},
    {0.58333333333333337034, 1.51480519480519482123},
    {0.75000000000000000000, 1.51480519480519482123},
    {0.91666666666666662966, 1.51480519480519482123},
    {0.10000000000000000555, -2.44649495951579298847},
    {0.29999999999999998890, -2.44649495951579298847},
    {0.50000000000000000000, -2.44649495951579298847},
    {0.69999999999999995559, -2.44649495951579298847},
    {0.90000000000000002220, -2.44649495951579298847},
    {0.12500000000000000000, 1.15583774250440907139},
    {0.37500000000000000000, 1.15583774250440907139},
    {0.62500000000000000000, 1.15583774250440907139},
    {0.87500000000000000000, 1.15583774250440907139},
    {0.16666666666666665741, -0.16272321428571429047},
    {0.50000000000000000000, -0.16272321428571429047},
    {0.83333333333333337034, -0.16272321428571429047},
    {0.25000000000000000000, 0.00423280423280423309},
    {0.75000000000000000000, 0.00423280423280423309},
    {0.50000000000000000000, -0.00000330687830687831},
};

template <>
std::vector<std::array<double, 3>> IntegrationRule<2, 0>::rule{
    {0.33333333333333331483, 0.33333333333333331483, 1.00000000000000000000},
};
template <>
std::vector<std::array<double, 3>> IntegrationRule<2, 1>::rule{
    {0.33333333333333331483, 0.33333333333333331483, 1.00000000000000000000},
};
template <>
std::vector<std::array<double, 3>> IntegrationRule<2, 2>::rule{
    {0.20000000000000001110, 0.20000000000000001110, 0.52083333333333337034},
    {0.59999999999999997780, 0.20000000000000001110, 0.52083333333333337034},
    {0.20000000000000001110, 0.59999999999999997780, 0.52083333333333337034},
    {0.33333333333333331483, 0.33333333333333331483, -0.56250000000000000000},
};
template <>
std::vector<std::array<double, 3>> IntegrationRule<2, 3>::rule{
    {0.20000000000000001110, 0.20000000000000001110, 0.52083333333333337034},
    {0.59999999999999997780, 0.20000000000000001110, 0.52083333333333337034},
    {0.20000000000000001110, 0.59999999999999997780, 0.52083333333333337034},
    {0.33333333333333331483, 0.33333333333333331483, -0.56250000000000000000},
};
template <>
std::vector<std::array<double, 3>> IntegrationRule<2, 4>::rule{
    {0.14285714285714284921, 0.14285714285714284921, 0.41684027777777776791},
    {0.42857142857142854764, 0.14285714285714284921, 0.41684027777777776791},
    {0.14285714285714284921, 0.42857142857142854764, 0.41684027777777776791},
    {0.71428571428571430157, 0.14285714285714284921, 0.41684027777777776791},
    {0.42857142857142854764, 0.42857142857142854764, 0.41684027777777776791},
    {0.14285714285714284921, 0.71428571428571430157, 0.41684027777777776791},
    {0.20000000000000001110, 0.20000000000000001110, -0.54253472222222220989},
    {0.59999999999999997780, 0.20000000000000001110, -0.54253472222222220989},
    {0.20000000000000001110, 0.59999999999999997780, -0.54253472222222220989},
    {0.33333333333333331483, 0.33333333333333331483, 0.12656249999999999445},
};
template <>
std::vector<std::array<double, 3>> IntegrationRule<2, 5>::rule{
    {0.14285714285714284921, 0.14285714285714284921, 0.41684027777777776791},
    {0.42857142857142854764, 0.14285714285714284921, 0.41684027777777776791},
    {0.14285714285714284921, 0.42857142857142854764, 0.41684027777777776791},
    {0.71428571428571430157, 0.14285714285714284921, 0.41684027777777776791},
    {0.42857142857142854764, 0.42857142857142854764, 0.41684027777777776791},
    {0.14285714285714284921, 0.71428571428571430157, 0.41684027777777776791},
    {0.20000000000000001110, 0.20000000000000001110, -0.54253472222222220989},
    {0.59999999999999997780, 0.20000000000000001110, -0.54253472222222220989},
    {0.20000000000000001110, 0.59999999999999997780, -0.54253472222222220989},
    {0.33333333333333331483, 0.33333333333333331483, 0.12656249999999999445},
};
template <>
std::vector<std::array<double, 3>> IntegrationRule<2, 6>::rule{
    {0.11111111111111110494, 0.11111111111111110494, 0.41189313616071426827},
    {0.33333333333333331483, 0.11111111111111110494, 0.41189313616071426827},
    {0.11111111111111110494, 0.33333333333333331483, 0.41189313616071426827},
    {0.55555555555555558023, 0.11111111111111110494, 0.41189313616071426827},
    {0.33333333333333331483, 0.33333333333333331483, 0.41189313616071426827},
    {0.11111111111111110494, 0.55555555555555558023, 0.41189313616071426827},
    {0.77777777777777779011, 0.11111111111111110494, 0.41189313616071426827},
    {0.55555555555555558023, 0.33333333333333331483, 0.41189313616071426827},
    {0.33333333333333331483, 0.55555555555555558023, 0.41189313616071426827},
    {0.11111111111111110494, 0.77777777777777779011, 0.41189313616071426827},
    {0.14285714285714284921, 0.14285714285714284921, -0.63828667534722227650},
    {0.42857142857142854764, 0.14285714285714284921, -0.63828667534722227650},
    {0.14285714285714284921, 0.42857142857142854764, -0.63828667534722227650},
    {0.71428571428571430157, 0.14285714285714284921, -0.63828667534722227650},
    {0.42857142857142854764, 0.42857142857142854764, -0.63828667534722227650},
    {0.14285714285714284921, 0.71428571428571430157, -0.63828667534722227650},
    {0.20000000000000001110, 0.20000000000000001110, 0.24220300099206348854},
    {0.59999999999999997780, 0.20000000000000001110, 0.24220300099206348854},
    {0.20000000000000001110, 0.59999999999999997780, 0.24220300099206348854},
    {0.33333333333333331483, 0.33333333333333331483, -0.01582031249999999931},
};
template <>
std::vector<std::array<double, 3>> IntegrationRule<2, 7>::rule{
    {0.11111111111111110494, 0.11111111111111110494, 0.41189313616071426827},
    {0.33333333333333331483, 0.11111111111111110494, 0.41189313616071426827},
    {0.11111111111111110494, 0.33333333333333331483, 0.41189313616071426827},
    {0.55555555555555558023, 0.11111111111111110494, 0.41189313616071426827},
    {0.33333333333333331483, 0.33333333333333331483, 0.41189313616071426827},
    {0.11111111111111110494, 0.55555555555555558023, 0.41189313616071426827},
    {0.77777777777777779011, 0.11111111111111110494, 0.41189313616071426827},
    {0.55555555555555558023, 0.33333333333333331483, 0.41189313616071426827},
    {0.33333333333333331483, 0.55555555555555558023, 0.41189313616071426827},
    {0.11111111111111110494, 0.77777777777777779011, 0.41189313616071426827},
    {0.14285714285714284921, 0.14285714285714284921, -0.63828667534722227650},
    {0.42857142857142854764, 0.14285714285714284921, -0.63828667534722227650},
    {0.14285714285714284921, 0.42857142857142854764, -0.63828667534722227650},
    {0.71428571428571430157, 0.14285714285714284921, -0.63828667534722227650},
    {0.42857142857142854764, 0.42857142857142854764, -0.63828667534722227650},
    {0.14285714285714284921, 0.71428571428571430157, -0.63828667534722227650},
    {0.20000000000000001110, 0.20000000000000001110, 0.24220300099206348854},
    {0.59999999999999997780, 0.20000000000000001110, 0.24220300099206348854},
    {0.20000000000000001110, 0.59999999999999997780, 0.24220300099206348854},
    {0.33333333333333331483, 0.33333333333333331483, -0.01582031249999999931},
};
template <>
std::vector<std::array<double, 3>> IntegrationRule<2, 8>::rule{
    {0.09090909090909091161, 0.09090909090909091161, 0.46149657126667215090},
    {0.27272727272727270709, 0.09090909090909091161, 0.46149657126667215090},
    {0.09090909090909091161, 0.27272727272727270709, 0.46149657126667215090},
    {0.45454545454545453032, 0.09090909090909091161, 0.46149657126667215090},
    {0.27272727272727270709, 0.27272727272727270709, 0.46149657126667215090},
    {0.09090909090909091161, 0.45454545454545453032, 0.46149657126667215090},
    {0.63636363636363635354, 0.09090909090909091161, 0.46149657126667215090},
    {0.45454545454545453032, 0.27272727272727270709, 0.46149657126667215090},
    {0.27272727272727270709, 0.45454545454545453032, 0.46149657126667215090},
    {0.09090909090909091161, 0.63636363636363635354, 0.46149657126667215090},
    {0.81818181818181823228, 0.09090909090909091161, 0.46149657126667215090},
    {0.63636363636363635354, 0.27272727272727270709, 0.46149657126667215090},
    {0.45454545454545453032, 0.45454545454545453032, 0.46149657126667215090},
    {0.27272727272727270709, 0.63636363636363635354, 0.46149657126667215090},
    {0.09090909090909091161, 0.81818181818181823228, 0.46149657126667215090},
    {0.11111111111111110494, 0.11111111111111110494, -0.83408360072544640573},
    {0.33333333333333331483, 0.11111111111111110494, -0.83408360072544640573},
    {0.11111111111111110494, 0.33333333333333331483, -0.83408360072544640573},
    {0.55555555555555558023, 0.11111111111111110494, -0.83408360072544640573},
    {0.33333333333333331483, 0.33333333333333331483, -0.83408360072544640573},
    {0.11111111111111110494, 0.55555555555555558023, -0.83408360072544640573},
    {0.77777777777777779011, 0.11111111111111110494, -0.83408360072544640573},
    {0.55555555555555558023, 0.33333333333333331483, -0.83408360072544640573},
    {0.33333333333333331483, 0.55555555555555558023, -0.83408360072544640573},
    {0.11111111111111110494, 0.77777777777777779011, -0.83408360072544640573},
    {0.14285714285714284921, 0.14285714285714284921, 0.43438954294463733019},
    {0.42857142857142854764, 0.14285714285714284921, 0.43438954294463733019},
    {0.14285714285714284921, 0.42857142857142854764, 0.43438954294463733019},
    {0.71428571428571430157, 0.14285714285714284921, 0.43438954294463733019},
    {0.42857142857142854764, 0.42857142857142854764, 0.43438954294463733019},
    {0.14285714285714284921, 0.71428571428571430157, 0.43438954294463733019},
    {0.20000000000000001110, 0.20000000000000001110, -0.06307369817501654041},
    {0.59999999999999997780, 0.20000000000000001110, -0.06307369817501654041},
    {0.20000000000000001110, 0.59999999999999997780, -0.06307369817501654041},
    {0.33333333333333331483, 0.33333333333333331483, 0.00127127511160714289},
};
template <>
std::vector<std::array<double, 3>> IntegrationRule<2, 9>::rule{
    {0.09090909090909091161, 0.09090909090909091161, 0.46149657126667215090},
    {0.27272727272727270709, 0.09090909090909091161, 0.46149657126667215090},
    {0.09090909090909091161, 0.27272727272727270709, 0.46149657126667215090},
    {0.45454545454545453032, 0.09090909090909091161, 0.46149657126667215090},
    {0.27272727272727270709, 0.27272727272727270709, 0.46149657126667215090},
    {0.09090909090909091161, 0.45454545454545453032, 0.46149657126667215090},
    {0.63636363636363635354, 0.09090909090909091161, 0.46149657126667215090},
    {0.45454545454545453032, 0.27272727272727270709, 0.46149657126667215090},
    {0.27272727272727270709, 0.45454545454545453032, 0.46149657126667215090},
    {0.09090909090909091161, 0.63636363636363635354, 0.46149657126667215090},
    {0.81818181818181823228, 0.09090909090909091161, 0.46149657126667215090},
    {0.63636363636363635354, 0.27272727272727270709, 0.46149657126667215090},
    {0.45454545454545453032, 0.45454545454545453032, 0.46149657126667215090},
    {0.27272727272727270709, 0.63636363636363635354, 0.46149657126667215090},
    {0.09090909090909091161, 0.81818181818181823228, 0.46149657126667215090},
    {0.11111111111111110494, 0.11111111111111110494, -0.83408360072544640573},
    {0.33333333333333331483, 0.11111111111111110494, -0.83408360072544640573},
    {0.11111111111111110494, 0.33333333333333331483, -0.83408360072544640573},
    {0.55555555555555558023, 0.11111111111111110494, -0.83408360072544640573},
    {0.33333333333333331483, 0.33333333333333331483, -0.83408360072544640573},
    {0.11111111111111110494, 0.55555555555555558023, -0.83408360072544640573},
    {0.77777777777777779011, 0.11111111111111110494, -0.83408360072544640573},
    {0.55555555555555558023, 0.33333333333333331483, -0.83408360072544640573},
    {0.33333333333333331483, 0.55555555555555558023, -0.83408360072544640573},
    {0.11111111111111110494, 0.77777777777777779011, -0.83408360072544640573},
    {0.14285714285714284921, 0.14285714285714284921, 0.43438954294463733019},
    {0.42857142857142854764, 0.14285714285714284921, 0.43438954294463733019},
    {0.14285714285714284921, 0.42857142857142854764, 0.43438954294463733019},
    {0.71428571428571430157, 0.14285714285714284921, 0.43438954294463733019},
    {0.42857142857142854764, 0.42857142857142854764, 0.43438954294463733019},
    {0.14285714285714284921, 0.71428571428571430157, 0.43438954294463733019},
    {0.20000000000000001110, 0.20000000000000001110, -0.06307369817501654041},
    {0.59999999999999997780, 0.20000000000000001110, -0.06307369817501654041},
    {0.20000000000000001110, 0.59999999999999997780, -0.06307369817501654041},
    {0.33333333333333331483, 0.33333333333333331483, 0.00127127511160714289},
};
template <>
std::vector<std::array<double, 3>> IntegrationRule<2, 10>::rule{
    {0.07692307692307692735, 0.07692307692307692735, 0.56211684239171255673},
    {0.23076923076923078204, 0.07692307692307692735, 0.56211684239171255673},
    {0.07692307692307692735, 0.23076923076923078204, 0.56211684239171255673},
    {0.38461538461538463674, 0.07692307692307692735, 0.56211684239171255673},
    {0.23076923076923078204, 0.23076923076923078204, 0.56211684239171255673},
    {0.07692307692307692735, 0.38461538461538463674, 0.56211684239171255673},
    {0.53846153846153843592, 0.07692307692307692735, 0.56211684239171255673},
    {0.38461538461538463674, 0.23076923076923078204, 0.56211684239171255673},
    {0.23076923076923078204, 0.38461538461538463674, 0.56211684239171255673},
    {0.07692307692307692735, 0.53846153846153843592, 0.56211684239171255673},
    {0.69230769230769229061, 0.07692307692307692735, 0.56211684239171255673},
    {0.53846153846153843592, 0.23076923076923078204, 0.56211684239171255673},
    {0.38461538461538463674, 0.38461538461538463674, 0.56211684239171255673},
    {0.23076923076923078204, 0.53846153846153843592, 0.56211684239171255673},
    {0.07692307692307692735, 0.69230769230769229061, 0.56211684239171255673},
    {0.84615384615384614531, 0.07692307692307692735, 0.56211684239171255673},
    {0.69230769230769229061, 0.23076923076923078204, 0.56211684239171255673},
    {0.53846153846153843592, 0.38461538461538463674, 0.56211684239171255673},
    {0.38461538461538463674, 0.53846153846153843592, 0.56211684239171255673},
    {0.23076923076923078204, 0.69230769230769229061, 0.56211684239171255673},
    {0.07692307692307692735, 0.84615384615384614531, 0.56211684239171255673},
    {0.09090909090909091161, 0.09090909090909091161, -1.16335594006806952727},
    {0.27272727272727270709, 0.09090909090909091161, -1.16335594006806952727},
    {0.09090909090909091161, 0.27272727272727270709, -1.16335594006806952727},
    {0.45454545454545453032, 0.09090909090909091161, -1.16335594006806952727},
    {0.27272727272727270709, 0.27272727272727270709, -1.16335594006806952727},
    {0.09090909090909091161, 0.45454545454545453032, -1.16335594006806952727},
    {0.63636363636363635354, 0.09090909090909091161, -1.16335594006806952727},
    {0.45454545454545453032, 0.27272727272727270709, -1.16335594006806952727},
    {0.27272727272727270709, 0.45454545454545453032, -1.16335594006806952727},
    {0.09090909090909091161, 0.63636363636363635354, -1.16335594006806952727},
    {0.81818181818181823228, 0.09090909090909091161, -1.16335594006806952727},
    {0.63636363636363635354, 0.27272727272727270709, -1.16335594006806952727},
    {0.45454545454545453032, 0.45454545454545453032, -1.16335594006806952727},
    {0.27272727272727270709, 0.63636363636363635354, -1.16335594006806952727},
    {0.09090909090909091161, 0.81818181818181823228, -1.16335594006806952727},
    {0.11111111111111110494, 0.11111111111111110494, 0.76773604157683139615},
    {0.33333333333333331483, 0.11111111111111110494, 0.76773604157683139615},
    {0.11111111111111110494, 0.33333333333333331483, 0.76773604157683139615},
    {0.55555555555555558023, 0.11111111111111110494, 0.76773604157683139615},
    {0.33333333333333331483, 0.33333333333333331483, 0.76773604157683139615},
    {0.11111111111111110494, 0.55555555555555558023, 0.76773604157683139615},
    {0.77777777777777779011, 0.11111111111111110494, 0.76773604157683139615},
    {0.55555555555555558023, 0.33333333333333331483, 0.76773604157683139615},
    {0.33333333333333331483, 0.55555555555555558023, 0.76773604157683139615},
    {0.11111111111111110494, 0.77777777777777779011, 0.76773604157683139615},
    {0.14285714285714284921, 0.14285714285714284921, -0.17737573003572690289},
    {0.42857142857142854764, 0.14285714285714284921, -0.17737573003572690289},
    {0.14285714285714284921, 0.42857142857142854764, -0.17737573003572690289},
    {0.71428571428571430157, 0.14285714285714284921, -0.17737573003572690289},
    {0.42857142857142854764, 0.42857142857142854764, -0.17737573003572690289},
    {0.14285714285714284921, 0.71428571428571430157, -0.17737573003572690289},
    {0.20000000000000001110, 0.20000000000000001110, 0.01095029482205148175},
    {0.59999999999999997780, 0.20000000000000001110, 0.01095029482205148175},
    {0.20000000000000001110, 0.59999999999999997780, 0.01095029482205148175},
    {0.33333333333333331483, 0.33333333333333331483, -0.00007150922502790179},
};
};  // namespace tools
