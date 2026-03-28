# OPCODE

# [3:0] type of layer
""" 
0000: CNN
0001: max pooling layer
oo10: avarage pool
0011: dense layer
future versions I hope will not use PS/Microblaze hence remianing bits
"""

# [11:4]   number of input  channels
# [19:12]  number of output channels
"""in dense layer input channels = output channels = 1"""


# [39:20] dimensions of input channel (height, width)
"""
[29:20]: height (max size = 1024), max size = 2^10 = 1024
[39:30]: width  (max size = 1024), max size = 2^10 = 1024

in the case that number of input values >1024 in future versions
it is possible to interpret number of input channels as [38:20]
"""

# [59:40] dimensions of output channel (height, width)
"""
[49:40]: height (max size = 1024), max size = 2^10 = 1024
[59:50]: width  (max size = 1024), max size = 2^10 = 1024

in the case that number of input values >1024 in future versions
it is possible to interpret number of input channels as [37:18]
"""

# [61:60] activation type (for future version. for now it is limited to ReLU)
"""
00: ReLU
Remaining to be dfined later
TODO: add code to pass leakyReLU parameter
"""

# [67:62] kernel dimension (height, width) (future versions may use non 3X3 or even rectangular kernels)
"""
[64:62] height
[67:65] width
"""
# [68] stride
"""
0: stride = 1
1: stride = 2
"""
# [69] padding
"""
0: padding = 0
1: padding = 1
"""

# [70] RESNET true?
"""
0: RESNET OFF
1: RESNET ON
"""

"""
Memory address is represented via data-table
512 Mb of DDR = 512*1024*1024 bits, to represent this we will need 29 bits
Each data-table value will hold (8) bits and a data table index will be passed in opcode

If this was not implemented opcode would become too large: 29*4 + others (smaller opcode requirement if )
A downside is some memory will remain unused (more memory wasted if data-table value increases)

I feel 8 bit should be adequate but TODO: optimize size of data-table value

calculation: (2^(n))^8 >= 2^29
=> 8n > 29 => n = 4
Hence id will be 4 bits long
"""
# [74:71] input address start id
# [78:75] output address start id
# [82:79] weight address start id
# [86:83] bias address start id

# [127:87] flags and other features

