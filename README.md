# Akira

Akira is a C library for Neural Networks.

<img src="/public/a-logo.png" alt="Akira logo" width=342 height=350/>

## Installation

If you are on Linux use can use the `install.sh` script.

The `install.sh` script will install - compile __akira__ and its dependencies alongside the dependencies for all the examples.

Clone this repository

```bash
git clone https://github.com/index-zer0/akira.git
```
Clone the dependencies (or use the `install.sh` script)
```bash
cd akira
git clone https://github.com/index-zer0/cmatrix.git
# To run the mnist example you will also need:
git clone https://github.com/takafumihoriuchi/MNIST_for_C/
# Edit the mnist.h file and add the path to the Mnist data
```


## Usage
Include __akira__ in your program

```C
#include "akira.h"
```
Follow the steps inside one of the examples

## Contributing
Pull requests are always welcome. 

For major changes, please open an issue first to discuss what you would like to change.


## License
[BSD-3-Clause](https://github.com/index-zer0/akira/blob/master/LICENSE)
