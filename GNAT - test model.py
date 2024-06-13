# General Neural Architechture Technology (GNAT)
import random
import time
import numpy as np

inputs = [[0, 1], [1, 0], [0, 0], [1, 1]]
expectedOutput = [1, 1, 0, 2]

numInputs = len(inputs[0])

layers = [[numInputs, False], [2, True], [3, False]]

numLayers = len(layers)

outputLen = layers[len(layers) - 1][0]

def process(param):
    largest = 0
    indice = 0
    for i in range(len(param)):
        if param[i] > largest:
            largest = param[i]
            indice = i
    return indice
        

def runGNAT(coefficients, input):
    start = input
    for i in range(numLayers - 1):
        x = 0
        end = [0 for _ in range(layers[i + 1][0])]
        for j in range(layers[i + 1][0]): #end loop
            for k in range(layers[i][0]): #start loop
                end[j] += start[k] * coefficients[i][x]
                x += 1
        if layers[i + 1][1]:
            for k in range(len(end)):
                end[k] *= coefficients[i][k + x]
                end[k] += coefficients[i][k + x + len(end)]
        start = end
    return end

def run(arg):
    success = 0
    solve = False
    correct = 0
    coeffs = arg
    for i in range(len(inputs)):
        output = runGNAT(coeffs, inputs[i])
        print(output)
        result = process(output)
        print(result)
        print("DONE")
        if result == expectedOutput[i]:
            correct += 1
            for j in range(outputLen):
                success += abs(output[result] - output[j]) / max(output[result], output[j], 0.1)
        else:
            for j in range(outputLen):
                success -= abs(output[result] - output[j]) / max(output[result], output[j], 0.1)
    if correct == len(inputs):
        solve = True
    print(str(correct) + " of " + str(len(inputs)) + " correct")
    
    success *= pow(correct, 3) / pow(len(inputs), 3) * 25
    
    return [solve, success]

if __name__ == "__main__":
    
            
    coef = [[0.5513513345248588, 0.19513345191511622, 0.7951601896065162, 0.9334205354500977, 0.028538397007365657, 1.0, 0.2248288312537491, 0.0], [0.6809739733857738, 0.005608055705567105, 0.19509515449475648, 0.9615542221182265, 0.0, 1.0, 0.6638721361663347, 0.7470762152802224], [0.02311369308800615, 0.24691101671748306, 0.9243406371144962, 0.8126118455014057, 0.9944090052501993, 0.6010531531454038, 0.0, 0.7321237625881325]]
    res = run(coef)
    print("SUCCESS: " + str(res[1]))
    print("SOLVED?: " + str(res[0]))
    
    