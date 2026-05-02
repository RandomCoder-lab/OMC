// Copyright Sovereign Lattice, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "Modules/ModuleInterface.h"

/**
 * OMNIcode Unreal Engine Plugin Module
 * Provides genetic circuit evolution functionality for game AI and procedural generation
 */
class FOmnimcodeModule : public IModuleInterface
{
public:
    /** IModuleInterface implementation */
    virtual void StartupModule() override;
    virtual void ShutdownModule() override;

private:
    /** Handle to the loaded omnimcode native library */
    void* LibraryHandle = nullptr;

    /** Load the native omnimcode library for the current platform */
    void LoadOmnimcodeLibrary();

    /** Unload the native omnimcode library */
    void UnloadOmnimcodeLibrary();
};
