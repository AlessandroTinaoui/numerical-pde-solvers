#include "DiffusionReactionDN.hpp"

enum class RelaxedData
{
  dirichlet,
  neumann
};

// Main function.
int main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

  const std::string  mesh_file_0 = "../mesh/mesh-problem-0.msh";
  const std::string  mesh_file_1 = "../mesh/mesh-problem-1.msh";
  const unsigned int degree      = 1;

  const auto alpha = [](const Point<DiffusionReactionDN::dim> &) { return 1.0; };
  const auto gamma = [](const Point<DiffusionReactionDN::dim> &) { return 1.0; };
  const auto f     = [](const Point<DiffusionReactionDN::dim> &) { return 1.0; };

  Functions::ZeroFunction<DiffusionReactionDN::dim> zero;
  DiffusionReactionDN::DirichletConditions          dirichlet_0;
  DiffusionReactionDN::DirichletConditions          dirichlet_1;
  DiffusionReactionDN::NeumannConditions            neumann_0;
  DiffusionReactionDN::NeumannConditions            neumann_1;

  dirichlet_0[0] = &zero;
  dirichlet_0[2] = &zero;
  dirichlet_0[3] = &zero;
  dirichlet_1[1] = &zero;
  dirichlet_1[2] = &zero;
  dirichlet_1[3] = &zero;

  DiffusionReactionDN problem_0(0, mesh_file_0, degree, 1, alpha, gamma, f, dirichlet_0, neumann_0);
  DiffusionReactionDN problem_1(1, mesh_file_1, degree, 0, alpha, gamma, f, dirichlet_1, neumann_1);

  problem_0.setup();
  problem_1.setup();

  std::cout << "Setup completed" << std::endl;

  const double       tolerance_increment = 1e-4;
  const unsigned int n_max_iter          = 100;

  double       solution_increment_norm = tolerance_increment + 1;
  unsigned int n_iter                  = 0;

  // Relaxation coefficient (1 = no relaxation).
  const double      lambda       = 0.25;
  const RelaxedData relaxed_data = RelaxedData::dirichlet;

  while (n_iter < n_max_iter && (n_iter < 2 || solution_increment_norm > tolerance_increment))
    {
      const Vector<double> previous_solution = problem_1.get_solution();

      problem_0.assemble();
      problem_0.apply_interface_dirichlet(problem_1);
      problem_0.solve();

      problem_1.assemble();
      const double neumann_increment = problem_1.apply_interface_neumann(problem_0, relaxed_data == RelaxedData::neumann ? lambda : 1.0);
      problem_1.solve();

      if (relaxed_data == RelaxedData::dirichlet)
        problem_1.apply_relaxation(previous_solution, lambda);

      if (relaxed_data == RelaxedData::dirichlet)
        {
          Vector<double> solution_increment = previous_solution;
          solution_increment -= problem_1.get_solution();
          solution_increment_norm = solution_increment.l2_norm();
        }
      else
        solution_increment_norm = neumann_increment;

      std::cout << "iteration " << n_iter << " - solution increment = " << solution_increment_norm << std::endl;

      problem_0.output(n_iter);
      problem_1.output(n_iter);

      ++n_iter;
    }

  return 0;
}
