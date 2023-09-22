# Install Ubuntu 20.04 (focal) with R base and R version management (https://rocker-project.org/images/versioned/r-ver.html)
FROM rocker/r-ver:4.3.0
# set the default repository to CRAN
RUN echo 'options(repos = c(CRAN = "https://cloud.r-project.org"))' >>"${R_HOME}/etc/Rprofile.site"
# sets the working directory for any RUN, CMD, ENTRYPOINT, COPY and ADD instructions
WORKDIR /context
# copy R scripts
COPY *.R ./
# install Ubuntu 20.04 libraries libssl-dev and libcurl4-openssl-dev required for the benchmarkme library
RUN apt-get update && \
    apt-get install -y libssl-dev \
    libcurl4-openssl-dev && \
    Rscript /context/install_packages.R
# set entrypoint
ENTRYPOINT  ["Rscript", "--max-ppsize=500000", "/context/GenomicBasedMachineLearning:1.0.R"]
# sets the working directory for any RUN, CMD, ENTRYPOINT, COPY and ADD instructions
WORKDIR /tmp
# set Docker image version
ADD VERSION .
