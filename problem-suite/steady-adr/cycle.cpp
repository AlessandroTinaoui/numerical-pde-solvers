#include <deal.II/base/convergence_table.h>

#include <iostream>

#include "SteadyADR.hpp"

static constexpr unsigned int dim = 2;

// Exact solution.
class ExactSolution : public Function<dim>
{
public:
  // Constructor.
  ExactSolution()
  {}

  // Evaluation.
  virtual double
  value(const Point<dim> &p, const unsigned int /*component*/ = 0) const override
  {
    return std::sin(2.0 * M_PI * p[0]) * std::sin(4.0 * M_PI * p[1]);
  }

  // Gradient evaluation.
  virtual Tensor<1, dim>
  gradient(const Point<dim> &p, const unsigned int /*component*/ = 0) const override
  {
    Tensor<1, dim> result;

    result[0] = 2.0 * M_PI * std::cos(2.0 * M_PI * p[0]) * std::sin(4.0 * M_PI * p[1]);
    result[1] = 4.0 * M_PI * std::sin(2.0 * M_PI * p[0]) * std::cos(4.0 * M_PI * p[1]);

    return result;
  }

  static constexpr double A = -4.0 / 15.0 * std::pow(0.5, 2.5);
};

// Main function.
int
main(int argc, char *argv[])
{
    Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);
    const unsigned int mpi_rank = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);

    //problem specific 
    ConvergenceTable table;
    std::ofstream convergence_file;
    if (mpi_rank == 0){
        convergence_file.open("convergence.csv");
        convergence_file << "h,eL2,eH1" << std::endl;
    }

    const std::vector<unsigned int> N_el_values = {5, 10, 20, 40};
    const ExactSolution exact_solution;

    // problem usual 
    const unsigned int degree = 2;

    // ADR
    const auto mu = [](const Point<dim> & /*p*/) { return 1.0; };
    const auto beta = [](const Point<dim> &) {
        Tensor<1, dim> value;
        value[0] = 0.0;
        value[1] = 0.0;
        return value;
    };
    const auto sigma = [](const Point<dim> & /*p*/) { return 1.0; };

    // forcing term
    const auto f = [](const Point<dim> &p) {
        return (20.0 * M_PI * M_PI + 1.0) * std::sin(2.0 * M_PI * p[0]) * std::sin(4.0 * M_PI * p[1]);
    };

    // boundary conditions
    SteadyADR<dim>::DirichletConditions dirichlet;
    SteadyADR<dim>::NeumannConditions neumann;

    // Dirichlet funcions
    Functions::ZeroFunction<dim> zero;
    dirichlet[0] = &zero;
    dirichlet[1] = &zero;
    dirichlet[2] = &zero;
    dirichlet[3] = &zero;

    for (const auto &N_el : N_el_values){
        const std::string mesh_file_name = "../mesh/mesh-square-" + std::to_string(N_el) + ".msh";

        SteadyADR<dim> problem(mesh_file_name, degree, mu, beta, sigma, f, dirichlet, neumann);

        problem.setup();
        problem.assemble();
        problem.solve();
        problem.output();

        // compute errors
        const double h = 1.0 / N_el;
        const double error_L2 = problem.compute_error(VectorTools::L2_norm, exact_solution);
        const double error_H1 = problem.compute_error(VectorTools::H1_norm, exact_solution);
        table.add_value("h", h);
        table.add_value("L2", error_L2);
        table.add_value("H1", error_H1);
        if (mpi_rank == 0)
            convergence_file << h << "," << error_L2 << "," << error_H1 << std::endl;
    }

    // Evaluate convergence rates.
    table.evaluate_all_convergence_rates(ConvergenceTable::reduction_rate_log2);
    table.set_scientific("L2", true);
    table.set_scientific("H1", true);
    if (mpi_rank == 0)
        table.write_text(std::cout);

    return 0;
}
