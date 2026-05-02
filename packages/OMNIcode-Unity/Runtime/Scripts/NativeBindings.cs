using System;
using System.Runtime.InteropServices;

namespace SovereignLattice.OMNIcode
{
    /// <summary>
    /// Low-level P/Invoke declarations for OMNIcode C FFI bindings
    /// </summary>
    internal static class NativeBindings
    {
        #region Library Loading

#if UNITY_EDITOR_WIN || UNITY_STANDALONE_WIN
        private const string LibraryName = "omnicode";
#elif UNITY_EDITOR_OSX || UNITY_STANDALONE_OSX
        private const string LibraryName = "libomnimcode";
#elif UNITY_EDITOR_LINUX || UNITY_STANDALONE_LINUX
        private const string LibraryName = "libomnimcode";
#else
        private const string LibraryName = "omnicode";
#endif

        #endregion

        #region Opaque Handles

        // Opaque pointers to Rust structures (never directly accessed from C#)
        public class CircuitHandle : SafeHandle
        {
            public CircuitHandle() : base(IntPtr.Zero, true) { }

            public override bool IsInvalid => handle == IntPtr.Zero;

            public override void Close()
            {
                if (!IsInvalid)
                {
                    omnicode_circuit_free(handle);
                }
            }

            protected override bool ReleaseHandle()
            {
                Close();
                return true;
            }
        }

        public class EvolverHandle : SafeHandle
        {
            public EvolverHandle() : base(IntPtr.Zero, true) { }

            public override bool IsInvalid => handle == IntPtr.Zero;

            public override void Close()
            {
                if (!IsInvalid)
                {
                    omnicode_evolver_free(handle);
                }
            }

            protected override bool ReleaseHandle()
            {
                Close();
                return true;
            }
        }

        #endregion

        #region FFI Functions

        /// <summary>
        /// Create a new circuit with specified number of inputs
        /// </summary>
        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern CircuitHandle omnicode_circuit_new(uint inputs);

        /// <summary>
        /// Evaluate circuit with given boolean inputs
        /// </summary>
        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern bool omnicode_circuit_eval(
            CircuitHandle circuit,
            [MarshalAs(UnmanagedType.LPArray, SizeParamIndex = 2)]
            bool[] inputs,
            ulong input_count
        );

        /// <summary>
        /// Free a circuit (called automatically by CircuitHandle via SafeHandle)
        /// </summary>
        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        private static extern void omnicode_circuit_free(IntPtr circuit);

        /// <summary>
        /// Create a new evolver for population-based evolution
        /// </summary>
        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern EvolverHandle omnicode_evolver_new(uint population_size);

        /// <summary>
        /// Run one generation of evolution
        /// </summary>
        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void omnicode_evolver_step(EvolverHandle evolver);

        /// <summary>
        /// Get the best circuit found so far
        /// </summary>
        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern CircuitHandle omnicode_evolver_best_circuit(EvolverHandle evolver);

        /// <summary>
        /// Get current generation number
        /// </summary>
        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern uint omnicode_evolver_generation(EvolverHandle evolver);

        /// <summary>
        /// Get best fitness found so far
        /// </summary>
        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern double omnicode_evolver_best_fitness(EvolverHandle evolver);

        /// <summary>
        /// Free an evolver (called automatically via SafeHandle)
        /// </summary>
        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        private static extern void omnicode_evolver_free(IntPtr evolver);

        /// <summary>
        /// Get version string
        /// </summary>
        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr omnicode_version();

        public static string GetVersion()
        {
            IntPtr versionPtr = omnicode_version();
            return Marshal.PtrToStringAnsi(versionPtr) ?? "1.0.0";
        }

        #endregion
    }
}
