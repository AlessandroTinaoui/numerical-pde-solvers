#include "SteadyADR.hpp"

static constexpr unsigned int dim = 2;

class DirichletFunction : public Function<dim>
{
public:
  double value(const Point<dim> &p, const unsigned int = 0) const override
  {
    return p[0] + p[1];
  }
};

int
main(int argc, char *argv[])
{
    Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

    //const unsigned int n_elements = 20;
    const unsigned int degree     = 1;

    // ADR
    const auto mu = [](const Point<dim> &) { return 1.0; };
    const auto beta = [](const Point<dim> &) {
        Tensor<1, dim> value;
        value[0] = 0.0;
        value[1] = 0.0;
        return value;
    };
    const auto sigma = [](const Point<dim> &) { return 0.0; };

    // forcing term
    const auto f = [](const Point<dim> &) { return -5.0; };

    // boundary conditions 
    SteadyADR<dim>::DirichletConditions dirichlet;
    SteadyADR<dim>::NeumannConditions neumann;

    // Dirichlet funcions
    Functions::ZeroFunction<dim> zero;
    DirichletFunction dirichlet_function;
    const auto neumann_function = [](const Point<dim> &p) { return p[1]; };

    dirichlet[0] = &dirichlet_function;
    dirichlet[1] = &dirichlet_function;

    neumann[2] = neumann_function;
    neumann[3] = neumann_function;
 
    // problem instance
    SteadyADR<dim> problem("../mesh/mesh-square-20.msh",
                            degree,
                            mu,
                            beta,
                            sigma,
                            f,
                            dirichlet,
                            neumann);

    // problem solution
    problem.setup();
    problem.assemble();
    problem.solve();
    problem.output();

    return 0;
}
