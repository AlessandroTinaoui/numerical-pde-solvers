#include "TransientADR.hpp"

int
main(int argc, char *argv[])
{
  constexpr unsigned int dim = TransientADR::dim;
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

  const auto mu = [](const Point<dim> &) { return 0.1; };
  const auto beta = [](const Point<dim> &) {
    Tensor<1, dim> value;
    return value;
  };
  const auto sigma = [](const Point<dim> &) { return 0.0; };
  const auto f = [](const Point<dim> &, double) { return 0.0; };
  const auto initial_condition = [](const Point<dim> &p, double) {
    return p[0] * (1.0 - p[0]) * p[1] * (1.0 - p[1]) *
           p[2] * (1.0 - p[2]);
  };

  TransientADR::BoundaryConditions dirichlet;
  TransientADR::BoundaryConditions neumann;

  TransientADR problem("../mesh/mesh-cube-10.msh",
                       1,
                       0.01,
                       1.0,
                       0.01,
                       mu,
                       beta,
                       sigma,
                       f,
                       initial_condition,
                       dirichlet,
                       neumann);

  problem.run();
}
