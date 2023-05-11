# STEROPES-WP4
    
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7925046.svg)](https://doi.org/10.5281/zenodo.7925046)

[![License: EUPL-1.2](https://img.shields.io/badge/License-EUPL%20v1.2-blue.svg)](https://joinup.ec.europa.eu/collection/eupl/eupl-text-eupl-12)

Command-line software for segmenting segment the vegetation from the RGB image

## Installation

To install and set up your command-line software and its dependencies, follow these steps:

1. Ensure that Python and [poetry](https://python-poetry.org/) are installed on your system. If not, please refer to the official documentation for installation instructions.

2. Clone the repository or download the source code for your command-line software.

```bash
git clone https://github.com/saberioon/vegRV.git
```

3. Navigate to the project directory in your terminal.

4. Run the following command to install the dependencies using Poetry:

```bash
poetry install
```


This command will create a virtual environment and install all the required dependencies specified in the `pyproject.toml` file.

For more information about Poetry and its usage, refer to the [official Poetry documentation](https://python-poetry.org/docs/).

5. Once the dependencies are installed successfully, you are ready to use your command-line software.

## Usage

To run your command-line software, follow these steps:

1. Make sure you are in the project directory in your terminal.

2. Activate the virtual environment created by Poetry by running the following command:

```bash
poetry shell
```
3. run the commandline software using this command 

```bash
python steropeswp4.py -i Input Folder -o Output Folder -c Colorspace {hsv, hls, yiq}
```

### options:
  -h, --help            show this help message and exit 
  -i INPUT, --input INPUT   input folder

  -o OUTPUT, --output OUTPUT output folder 

  -c COLORSPACE, --colorspace COLORSPACE  colorspace hsv, hls, yiq 

## Additional Notes

- It is recommended to always run your command-line software within the Poetry virtual environment to ensure proper dependency management and avoid conflicts with other packages installed on your system.

- keep the input folder and output folder in seperate directory.

## License

this software is released under EUROPEAN UNION PUBLIC LICENCE v. 1.2


## Acknowledgements

This work was supported by the European Union’s Horizon H2020 research and innovation European Joint Programme Cofund on Agricultural Soil Management (EJP-SOIL grant number 862695) and was carried out in the framework of the STEROPES of EJP-SOIL

[![SteropesLogo](https://github.com/saberioon/vegRV/blob/main/figs/steropesLogo.jpg)]()
