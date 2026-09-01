#include "DiffusionReactionDD.hpp"

// Main function.
int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

  const std::string mesh_file_0 = "../mesh/mesh-problem-0.msh";
  const std::string mesh_file_1 = "../mesh/mesh-problem-1.msh";
  const unsigned int degree = 1;

  const auto alpha = [](const Point<DiffusionReactionDD::dim> &) {
    return 1.0;
  };
  const auto gamma = [](const Point<DiffusionReactionDD::dim> &) {
    return 1.0;
  };
  const auto f = [](const Point<DiffusionReactionDD::dim> &) {
    return 1.0;
  };

  Functions::ZeroFunction<DiffusionReactionDD::dim> zero;
  DiffusionReactionDD::DirichletConditions dirichlet_0;
  DiffusionReactionDD::DirichletConditions dirichlet_1;
  DiffusionReactionDD::NeumannConditions neumann_0;
  DiffusionReactionDD::NeumannConditions neumann_1;

  dirichlet_0[0] = &zero;
  dirichlet_0[2] = &zero;
  dirichlet_0[3] = &zero;
  dirichlet_1[1] = &zero;
  dirichlet_1[2] = &zero;
  dirichlet_1[3] = &zero;

  DiffusionReactionDD problem_0(0, // subdomain id
                                mesh_file_0, // mesh file name
                                degree, // polynomial degree    
                                1,// interface boundary id
                                alpha, // diffusion coefficient 
                                gamma, // reaction coefficient
                                f, // forcing term
                                dirichlet_0, // Dirichlet boundary conditions
                                neumann_0); // Neumann boundary conditions
  DiffusionReactionDD problem_1(1, // subdomain id
                                mesh_file_1, // mesh file name
                                degree, // polynomial degree
                                0, // interface boundary id
                                alpha, // diffusion coefficient
                                gamma, // reaction coefficient
                                f, // forcing term
                                dirichlet_1, // Dirichlet boundary conditions
                                neumann_1); // Neumann boundary conditions

  problem_0.setup();
  problem_1.setup();

  std::cout << "Setup completed" << std::endl;

  const double       tolerance_correction = 1e-4;
  const unsigned int n_max_iter           = 100;
  const double       lambda               = 0.5;
  const double       smoothing            = 0.1;

  double       correction_norm = tolerance_correction + 1.0;
  unsigned int n_iter          = 0;

  while (n_iter < n_max_iter && correction_norm > tolerance_correction)
    {
      problem_0.assemble();
      problem_0.apply_interface_dirichlet(problem_1);
      problem_0.solve();

      problem_1.assemble();
      problem_1.apply_interface_dirichlet(problem_0);
      problem_1.solve();

      correction_norm =
        problem_0.apply_interface_correction(problem_1, lambda, smoothing);

      std::cout << "iteration " << n_iter
                << " - interface correction = " << correction_norm
                << std::endl;

      problem_0.output(n_iter);
      problem_1.output(n_iter);

      ++n_iter;
    }

  return 0;
}
