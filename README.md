# before-effects

Small rust project for learning something about OpenCL.
First argument to the program is a source image file. 
After that you can list different effects separated by spaces.
Currently supported effects are:

| Argument                     | Explanation |
| ---------------------------- | ----------- |
| ```invert```                 | Inverts colors of the whole image. |
| ```greyscale```              | Grey scales the whole image. |
| ```contrast=[Intensity]```   | Adjusts the contrast. Intensity is a number between 0 and 1 with 0.5 not having no effect at all. |
| ```brightness=[Intensity]``` | Adjusts the brightness. Intensity is a number between 0 and 1 with 0.5 not having no effect at all. |
| ```chswap=[r,g,b,a]```       | Swaps color channels from RGB to the given order. If there are only 3 arguments, the alpha channel will stay untouched. |

## Debug output

Debug output can be turned on by setting the DEBUG environment
variable to any value (it just has to exist). For example:

```$ DEBUG= before-effects test.png invert``` 