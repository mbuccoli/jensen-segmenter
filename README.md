# Jensen Segmenter



## Description ##
This script contains an implementation of the algorithm proposed in "Multiple scale music segmentation using rhythm, timbre, and harmony" by Kristoffer Jensen for music segmentation (boundary detection). It uses [numpy](https://www.numpy.org) and [scipy](https://www.scipy.org) to compute the boundaries as the shortest path of a graph whose adjacency matrix is computed from the self-similarity matrix. 

The script follows the grammar used in the [Music Structure Analysis Framework](https://github.com/urinieto/msaf/) by Oriol Nieto.


## Installing the script #
The script can be simply copied in the folder in which you need it, or its directory can be added in the Python path.

## Using the script ##
The script has a main function `segment` that can be called with any matrix of features:

	from JensenSegmenter import segment
	F= #some feature extraction method
	est_idxs, _=segment(F)
    
The result `est_idxs` can be used with the [mir_eval](https://github.com/craffel/mir_eval) framework for evaluation.

## Requirements ##

* Python 3.x
* Numpy
* Scipy

## References ##

Jensen, K., (2007). Multiple scale music segmentation using rhythm, timbre, and harmony. In EURASIP Journal on Advances in
## Credits ##

Created by [Michele Buccoli](http://home.deib.polimi.it/buccoli) (<michele.buccoli@polimi.it>).