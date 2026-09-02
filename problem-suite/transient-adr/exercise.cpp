#include "TransientADR.hpp"

int main(int argc, char *argv[])
{
  constexpr unsigned int           dim = 3;
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

  const std::string  mesh_file_name = "../mesh/mesh-cube-10.msh";
  const unsigned int degree         = 1;
  const double       final_time     = 1.0;
  const double       theta          = 1.0;
  const double       delta_t        = 0.05;

  const auto mu   = [](const Point<dim> &) { return 0.1; };
  const auto beta = [](const Point<dim> &) {
    Tensor<1, dim> value;
    value[0] = 0.0;
    value[1] = 0.0;
    value[2] = 0.0;
    return value;
  };
  const auto sigma             = [](const Point<dim> &) { return 0.0; };
  const auto f                 = [](const Point<dim> &, double) { return 0.0; };
  const auto initial_condition = [](const Point<dim> &p, double) { return p[0] * (1.0 - p[0]) * p[1] * (1.0 - p[1]) * p[2] * (1.0 - p[2]); };

  TransientADR::BoundaryConditions dirichlet;
  TransientADR::BoundaryConditions neumann;

  TransientADR problem(mesh_file_name, degree, final_time, theta, delta_t, mu, beta, sigma, f, initial_condition, dirichlet, neumann);

  problem.run();
}
