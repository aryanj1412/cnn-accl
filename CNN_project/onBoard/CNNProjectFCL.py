# contains code for fully connected layer

import numpy as np

# funtion to process one fcl TODO: control bit length, TODO: add try-except to check if dimensions are correct
def fcl_single_layer(weights, bias, data, output_size, input_size):
    output = np.zeros(shape=output_size)
    for i in range(output_size):


        sum = 0
        for j in range(input_size):
            sum = sum + weights[i*input_size+j]*data[j] + bias[j]
        
        #ReLU
        if(sum>0):
            output[i] = sum
        else:
            output[i] = 0

    return output

# control contains info regarding number of layers, info about layer(size of input channel).
# control is a list; final_output_size is last element of control

# overall controller. TODO: make this more efficient. No point loading all the weights and biases in one go
def fcl_total(control:list, weights_total, bias_total, data):
    output = data
    # keeps track of which weight/bias is passed
    index = 0
    for i in range(len(control)-1):
        output=fcl_single_layer(weights=weights_total[index:index+control[i]*control[i+1]], 
                                bias=bias_total[index:index+control[i]*control[i+1]], 
                                data=output, output_size=control[i+1], input_size=control[i])
        index = index+control[i]*control[i+1]

    return output


# dummy test block
control = [2,2,2]
weights = np.array([1,1,1,1,1,1,1,1])
bias    = np.zeros(8)
data    = np.array([1.3,1.3])

print(fcl_total(control=control, weights_total=weights, bias_total=bias, data=data))