// Copyright Sovereign Lattice, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "UObject/NoExportTypes.h"
#include "OmnimcodeCircuit.generated.h"

// Forward declarations of C FFI types
typedef void* OmnimcodeCircuitHandle;
typedef void* OmnimcodeEvolverHandle;

/**
 * Unreal Engine wrapper for OMNIcode Circuit
 * Represents a logical circuit that can be evaluated and evolved
 */
UCLASS(BlueprintType)
class OMNIMCODE_API UOmnimcodeCircuit : public UObject
{
    GENERATED_BODY()

public:
    /**
     * Create a new circuit with specified number of inputs
     * @param NumInputs Number of boolean inputs (2-8 recommended)
     */
    UFUNCTION(BlueprintCallable, Category = "OMNIcode")
    void Create(int32 NumInputs);

    /**
     * Evaluate the circuit with given boolean inputs
     * @param Inputs Boolean input values
     * @return Output value
     */
    UFUNCTION(BlueprintCallable, Category = "OMNIcode")
    bool Evaluate(const TArray<bool>& Inputs);

    /**
     * Get number of gates in the circuit
     */
    UFUNCTION(BlueprintCallable, Category = "OMNIcode")
    int32 GetGateCount() const;

    /**
     * Get circuit complexity metric
     */
    UFUNCTION(BlueprintCallable, Category = "OMNIcode")
    int32 GetComplexity() const;

    /**
     * Destructor - clean up native resources
     */
    virtual ~UOmnimcodeCircuit();

private:
    /** Native FFI handle to the C circuit object */
    OmnimcodeCircuitHandle CircuitHandle = nullptr;

    /** Number of inputs for this circuit */
    int32 NumInputs = 0;
};

/**
 * Unreal Engine wrapper for OMNIcode Evolver
 * Evolves circuits to match boolean function specifications
 */
UCLASS(BlueprintType)
class OMNIMCODE_API UOmnimcodeEvolver : public UObject
{
    GENERATED_BODY()

public:
    /**
     * Create a new evolver for evolving circuits
     * @param PopulationSize Number of circuits in population (32-256)
     */
    UFUNCTION(BlueprintCallable, Category = "OMNIcode")
    void Create(int32 PopulationSize);

    /**
     * Run one generation of evolution
     * @param TestCases Boolean test case pairs (input → expected output)
     * @param NumGenerations Number of evolution steps
     */
    UFUNCTION(BlueprintCallable, Category = "OMNIcode")
    void Evolve(const TArray<FString>& TestCases, int32 NumGenerations);

    /**
     * Get the best circuit from current population
     */
    UFUNCTION(BlueprintCallable, Category = "OMNIcode")
    UOmnimcodeCircuit* GetBestCircuit();

    /**
     * Destructor - clean up native resources
     */
    virtual ~UOmnimcodeEvolver();

private:
    /** Native FFI handle to the C evolver object */
    OmnimcodeEvolverHandle EvolverHandle = nullptr;
};
