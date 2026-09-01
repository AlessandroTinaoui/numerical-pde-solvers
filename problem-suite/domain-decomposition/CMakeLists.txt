cmake_minimum_required(VERSION 3.12)
project(domain_decomposition LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
set(CMAKE_CXX_COMPILER "${MPI_CXX_COMPILER}")

find_package(deal.II 9.3.1 REQUIRED
  HINTS ${DEAL_II_DIR} $ENV{DEAL_II_DIR} $ENV{mkDealiiPrefix})
deal_ii_initialize_cached_variables()

add_executable(exercise-dn exercise-dn.cpp DiffusionReactionDN.cpp)
deal_ii_setup_target(exercise-dn)

add_executable(exercise-dd exercise-dd.cpp DiffusionReactionDD.cpp)
deal_ii_setup_target(exercise-dd)

add_custom_target(data-cleanup
  COMMAND ${CMAKE_COMMAND}
    -DOUTPUT_DIR=${CMAKE_CURRENT_BINARY_DIR}
    -P ${CMAKE_CURRENT_SOURCE_DIR}/../cmake/DataCleanup.cmake
  COMMENT "Removing generated solution data")
