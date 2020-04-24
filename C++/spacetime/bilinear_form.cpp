
#include "bilinear_form.hpp"

namespace spacetime {

// Template specializations for faster compile times.
template class BilinearForm<Time::MassOperator, space::StiffnessOperator,
                            Time::OrthonormalWaveletFn,
                            Time::OrthonormalWaveletFn>;
template class BilinearForm<Time::TransportOperator, space::MassOperator,
                            Time::ThreePointWaveletFn,
                            Time::OrthonormalWaveletFn>;
template class BilinearForm<Time::MassOperator, space::StiffnessOperator,
                            Time::ThreePointWaveletFn,
                            Time::OrthonormalWaveletFn>;
template class BilinearForm<Time::ZeroEvalOperator, space::MassOperator,
                            Time::ThreePointWaveletFn,
                            Time::ThreePointWaveletFn>;
template class SymmetricBilinearForm<
    Time::MassOperator, space::StiffnessOperator, Time::OrthonormalWaveletFn>;
template class SymmetricBilinearForm<
    Time::ZeroEvalOperator, space::MassOperator, Time::ThreePointWaveletFn>;
template class BlockDiagonalBilinearForm<
    space::DirectInverse<space::StiffnessOperator>, Time::OrthonormalWaveletFn,
    Time::OrthonormalWaveletFn>;
template class BlockDiagonalBilinearForm<
    space::XPreconditionerOperator<space::DirectInverse>,
    Time::ThreePointWaveletFn, Time::ThreePointWaveletFn>;
template class BlockDiagonalBilinearForm<
    space::MultigridPreconditioner<space::StiffnessOperator>,
    Time::OrthonormalWaveletFn, Time::OrthonormalWaveletFn>;
template class BlockDiagonalBilinearForm<
    space::XPreconditionerOperator<space::MultigridPreconditioner>,
    Time::ThreePointWaveletFn, Time::ThreePointWaveletFn>;

}  // namespace spacetime
