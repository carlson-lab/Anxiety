### Setting up Singularity

Our work makes use of a singularity file to simplify replicability of our work. 
This singularity file creates a virtual environment that replicates our developoment
environment exactly. The only package unable to be installed with the singularity package
is our lpne directory which is not publicly published on pip. Here, I'll outline how to 
install singularity and run our environment.

Singularity must be installed on a linux system. If you are running windows, you can do this by
making use of the windows subsystem for linux in windows terminal.

Once in your linux enviornment, run the following code in your home directory:

```sudo apt-get update && \
sudo apt-get install -y build-essential \
libseccomp-dev pkg-config squashfs-tools cryptsetup

sudo rm -r /usr/local/go

export VERSION=1.13.15 OS=linux ARCH=amd64  # change this as you need

wget -O /tmp/go${VERSION}.${OS}-${ARCH}.tar.gz https://dl.google.com/go/go${VERSION}.${OS}-${ARCH}.tar.gz && \
sudo tar -C /usr/local -xzf /tmp/go${VERSION}.${OS}-${ARCH}.tar.gz

echo 'export GOPATH=${HOME}/go' >> ~/.bashrc && \
echo 'export PATH=/usr/local/go/bin:${PATH}:${GOPATH}/bin' >> ~/.bashrc && \
source ~/.bashrc

curl -sfL https://install.goreleaser.com/github.com/golangci/golangci-lint.sh |
sh -s -- -b $(go env GOPATH)/bin v1.21.0

mkdir -p ${GOPATH}/src/github.com/sylabs && \
cd ${GOPATH}/src/github.com/sylabs && \
git clone https://github.com/sylabs/singularity.git && \
cd singularity

git checkout v3.6.3

cd ${GOPATH}/src/github.com/sylabs/singularity && \
./mconfig && \
cd ./builddir && \
make && \
sudo make install

singularity version```

After this you are ready to create your singularity container. 

Build the anxiety container using the following command:

```sudo singularity build cpne_anxiety.sif cpne_anxiety.def```

### Creating the python environment

The singularity container can be used with Duke Compute Cluster by referencing my other
tutorial. However, if you want to run this locally or on another system, use the command:

```singularity shell -i cpne_anxiety.sif```

From here, install the version of lpne that was used to train the model by navigating to ./lpne/ 
and run:

```pip install .```

From there you should be ready to do a projection

### Demo Projection

I've included the Clock-19 projection data as a demo and the orignal projection I ran in my environment.
Run the code there and check that the individual network activity lines up with what is loaded from the
existing projection document.

You can use this as a template for projecting into future datasets by replacing the loaded datafile.

#### ENSURE ALL FEATURES ARE GENERATED USING SAVEFEATURES 1.2 AND THAT POWER FEATURES ARE SCALED BY 10