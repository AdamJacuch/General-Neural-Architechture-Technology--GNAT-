# General Neural Architechture Technology (GNAT)
import random
import time
import numpy as np

batches = 5
iters = 5
length = 1000

inputs = [[0, 1], [1, 0], [0, 0], [1, 1], [1, 1]]
expectedOutput = [1, 1, 0, 2, 2] #[0, 1, 1, 0, 1, 0, 1, 0]

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

def modify(arr, amt):
    arr_np = np.array(arr)  # Convert input list to NumPy array
    modified_arr = arr_np + np.random.uniform(-amt, amt, size=arr_np.shape)
    modified_arr = np.clip(modified_arr, 0, 1)  # Ensure values are within [0, 1]
    return modified_arr.tolist()

def rng():
    return random.random()

def run(arg):
    success = 0
    solve = False
    correct = 0
    coeffs = arg
    
    inputLength = len(inputs)
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
    if correct == inputLength:
        solve = True
    print(str(correct) + " of " + str(inputLength) + " correct")
    
    success *= pow(correct, 3) / pow(inputLength, 3) * 25
    
    return [solve, success]

if __name__ == "__main__":
    firstIteration = True
    gene = []
    for _ in range(iters):
        SCORES = []
        GENES = []
        for _ in range(batches):
            MAX = -999
            PASSED_GENE = initCoeffs()
            
            iterations = 0
            firstGen = True
            gene = initCoeffs()
            while iterations < length:
                maxSuccess = -999
                for n in range(100):
                    iterations += 1
                    if firstIteration:
                        if firstGen:
                            coef = initCoeffs()
                        else:
                            coef = modify(gene, 0.1)
                    else:
                        if firstGen:
                            coef = modify(tempGene, 0.2)
                        else:
                            coef = modify(gene, 0.1)
                    res = run(coef)
                    print("SUCCESS: " + str(res[1]))
                    if res[1] > maxSuccess:
                        maxSuccess = res[1]
                        gene = coef
                    if res[1] > MAX:
                        MAX = res[1]
                        PASSED_GENE = coef
                firstGen = False
            SCORES.append(MAX)
            GENES.append(PASSED_GENE)
        firstIteration = False
        tempBest = -999
        tempGene = []
        for x in range(len(SCORES)):
            if SCORES[x] > tempBest:
                tempBest = SCORES[x]
                tempGene = GENES[x]
    print("DONE IN " + str(iterations) + " ITERATIONS")
    print("BEST: " + str(tempBest))
    print("-------------------------GENE-------------------------")
    print(tempGene)
    