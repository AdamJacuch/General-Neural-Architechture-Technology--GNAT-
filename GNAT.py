# General Neural Architechture Technology (GNAT)
import random
import time
import numpy as np

inputs = [[0, 1, 0], [1, 0, 1], [1, 1, 0], [0, 0, 1], [0, 1, 1], [1, 0, 0], [1, 1, 1], [0, 0, 0]]
expectedOutput = [1, 2, 2, 1, 2, 1, 3, 0]#[0, 1, 1, 0, 1, 0, 1, 0]

numInputs = len(inputs[0])

layers = [[numInputs, False], [2, True], [4, False]]

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

def initCoeffs():
    longest = 0
    for i in range(numLayers - 1):
        temp = layers[i][0]
        temp *= layers[i + 1][0]
        if layers[i + 1][1]:
            temp += layers[i + 1][0] * 2
        if temp > longest:
            longest = temp
    return [[rng() for _ in range(longest)] for _ in range(numLayers)]

def modify(arr):
    arr_np = np.array(arr)  # Convert input list to NumPy array
    modified_arr = arr_np + np.random.uniform(-0.1, 0.1, size=arr_np.shape)
    modified_arr = np.clip(modified_arr, 0, 1)  # Ensure values are within [0, 1]
    return modified_arr.tolist()

def rng():
    return random.random()

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
    
    success *= correct
    
    return [solve, success]

if __name__ == "__main__":
    solved = False
    iterations = 0
    firstGen = True
    gene = initCoeffs()
    MAX = -999
    while not solved and iterations < 15000:
        maxSuccess = -999
        for n in range(100):
            iterations += 1
            if firstGen:
                coef = initCoeffs()
            else:
                coef = modify(gene)
            res = run(coef)
            print("SUCCESS: " + str(res[1]))
            if res[0]:
                solved = True
                break
            if res[1] > maxSuccess:
                maxSuccess = res[1]
                gene = coef
                if maxSuccess > MAX:
                    MAX = maxSuccess
                
        firstGen = False
    print("DONE IN " + str(iterations) + " ITERATIONS")
    print("SUCCESS: " + str(MAX))
    print("-------------------------GENE-------------------------")
    print(gene)
    