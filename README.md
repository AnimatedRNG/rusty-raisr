# rusty-raisr

This tool upscales images using Google's RAISR algorithm.

_TODO_: Implement the training part of the algorithm. Currently, this tool uses a filterbank that is exported from movehand's [Python implementation](https://github.com/movehand/raisr).

# Usage

```
USAGE:
    rusty_raisr [OPTIONS] <input> <output>

FLAGS:
    -h, --help       Prints help information
    -V, --version    Prints version information

OPTIONS:
    -d <FILE>        Name of filter image to write
    -f <FILE>        Filterbank to use [default: filters/filterbank]

ARGS:
    <input>     Input image
    <output>    Output image
```

The "filter image" is a useful tool for visualizing which filter will be used per pixel.

![nearest neighbor, bicubic, and RAISR](img/comparison.png)

(from left to right: nearest neighbor, bicubic, and RAISR)
