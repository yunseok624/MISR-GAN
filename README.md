# MISR-GAN

## Model

## Datasets

### PROBA-V

### S2-NAIP

## Training
To train a model on this dataset, run the following command, with the desired configuration file:

`python -m ssr.train -opt ssr/options/*.yml` 

There are several sample configuration files in `ssr/options/`. Make sure the configuration file specifies 
correct paths to your downloaded data, the desired number of low-resolution input images, model parameters, 
and pretrained weights (if applicable).

## Testing
To evaluate the model on a test set run the following command, with the desired configuration file:

`python -m ssr.test -opt ssr/options/*.yml`

## Results

## Acknowledgements
Thanks to these codebases for foundational Super-Resolution code and inspiration:

[BasicSR](https://github.com/XPixelGroup/BasicSR/tree/master})

[ESRGAN](https://github.com/xinntao/ESRGAN/tree/master)

## Contact
If you have any questions, please email `yunseok.park@skoltech.ru` or [open an issue](https://github.com/yunseok624/MISR-GAN/issues/new).
