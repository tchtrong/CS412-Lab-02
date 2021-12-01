#ifndef SIFTARGS_HPP
#define SIFTARGS_HPP

struct SIFTArgs
{
    int nfeatures = 0;
    int nOctaveLayers = 3;
    double contrastThreshold = 0.04;
    double edgeThreshold = 10;
    double sigma = 1.6;
};

#endif