// Copyright Sovereign Lattice, Inc. All Rights Reserved.

using UnrealBuildTool;
using System.IO;

public class OMNIcode : ModuleRules
{
    public OMNIcode(ReadOnlyTargetRules Target) : base(Target)
    {
        PCHUsage = ModuleRules.PCHUsageMode.UseExplicitOrSharedPCHs;
        
        PublicIncludePaths.AddRange(
            new string[] {
                Path.Combine(ModuleDirectory, "Public"),
            }
        );

        PrivateIncludePaths.AddRange(
            new string[] {
                Path.Combine(ModuleDirectory, "Private"),
                Path.Combine(ModuleDirectory, "../ThirdParty/OMNIcode"),
            }
        );

        PublicDependencyModuleNames.AddRange(
            new string[] {
                "Core",
            }
        );

        PrivateDependencyModuleNames.AddRange(
            new string[] {
                "CoreUObject",
                "Engine",
            }
        );

        // Add omnimcode native library
        string OmnimcodePath = Path.Combine(ModuleDirectory, "../ThirdParty/OMNIcode");
        
        if (Target.Platform == UnrealTargetPlatform.Win64)
        {
            string LibPath = Path.Combine(OmnimcodePath, "Binaries", "Win64");
            PublicAdditionalLibraries.Add(Path.Combine(LibPath, "omnimcode.lib"));
            PublicDelayLoadDLLs.Add("omnimcode.dll");
            RuntimeDependencies.Add(Path.Combine(LibPath, "omnimcode.dll"));
        }
        else if (Target.Platform == UnrealTargetPlatform.Linux)
        {
            string LibPath = Path.Combine(OmnimcodePath, "Binaries", "Linux");
            PublicAdditionalLibraries.Add(Path.Combine(LibPath, "libomnimcode.so"));
        }
        else if (Target.Platform == UnrealTargetPlatform.Mac)
        {
            string LibPath = Path.Combine(OmnimcodePath, "Binaries", "Mac");
            PublicAdditionalLibraries.Add(Path.Combine(LibPath, "libomnimcode.dylib"));
        }

        PublicIncludePaths.Add(Path.Combine(OmnimcodePath, "Include"));
    }
}
