// omnimcode-gdextension/src/omnimcode_extension.cpp
// GDExtension binding for OMNIcode

#include "omnimcode.h"
#include <godot_cpp/classes/global_constants.hpp>
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/godot.hpp>
#include <godot_cpp/classes/ref_counted.hpp>

using namespace godot;

class OmnimcodeVMRef : public RefCounted {
    GDCLASS(OmnimcodeVMRef, RefCounted);

private:
    OmnimcodeVM* vm_handle;

public:
    OmnimcodeVMRef() : vm_handle(nullptr) {}
    
    ~OmnimcodeVMRef() {
        if (vm_handle) {
            omnicode_vm_free(vm_handle);
            vm_handle = nullptr;
        }
    }

    Error initialize() {
        vm_handle = omnicode_vm_new();
        if (!vm_handle) {
            return ERR_CANT_CREATE;
        }
        return OK;
    }

    int execute(const String& source) {
        if (!vm_handle) return -1;
        CharString cs = source.utf8();
        return omnicode_vm_execute(vm_handle, cs.get_data());
    }

    int reset() {
        if (!vm_handle) return -1;
        return omnicode_vm_reset(vm_handle);
    }

protected:
    static void _bind_methods() {
        ClassDB::bind_method(D_METHOD("execute", "source"), &OmnimcodeVMRef::execute);
        ClassDB::bind_method(D_METHOD("reset"), &OmnimcodeVMRef::reset);
    }
};

class OmnimcodeCircuitRef : public RefCounted {
    GDCLASS(OmnimcodeCircuitRef, RefCounted);

private:
    OmnimcodeCircuit* circuit_handle;
    int input_count;

public:
    OmnimcodeCircuitRef() : circuit_handle(nullptr), input_count(0) {}
    
    ~OmnimcodeCircuitRef() {
        if (circuit_handle) {
            omnicode_circuit_free(circuit_handle);
            circuit_handle = nullptr;
        }
    }

    Error initialize(int inputs) {
        circuit_handle = omnicode_circuit_new(inputs);
        if (!circuit_handle) {
            return ERR_CANT_CREATE;
        }
        input_count = inputs;
        return OK;
    }

    bool evaluate(const PackedByteArray& inputs) {
        if (!circuit_handle || inputs.size() != input_count) {
            return false;
        }
        const bool* data = reinterpret_cast<const bool*>(inputs.ptr());
        return omnicode_circuit_eval(circuit_handle, data, input_count);
    }

protected:
    static void _bind_methods() {
        ClassDB::bind_method(D_METHOD("evaluate", "inputs"), &OmnimcodeCircuitRef::evaluate);
    }
};

class OmnimcodeEvolverRef : public RefCounted {
    GDCLASS(OmnimcodeEvolverRef, RefCounted);

private:
    OmnimcodeEvolver* evolver_handle;

public:
    OmnimcodeEvolverRef() : evolver_handle(nullptr) {}
    
    ~OmnimcodeEvolverRef() {
        if (evolver_handle) {
            omnicode_evolver_free(evolver_handle);
            evolver_handle = nullptr;
        }
    }

    Error initialize(int population_size) {
        evolver_handle = omnicode_evolver_new(population_size);
        if (!evolver_handle) {
            return ERR_CANT_CREATE;
        }
        return OK;
    }

    void step() {
        if (evolver_handle) {
            omnicode_evolver_step(evolver_handle);
        }
    }

    int get_generation() const {
        if (!evolver_handle) return 0;
        return omnicode_evolver_generation(evolver_handle);
    }

    double get_best_fitness() const {
        if (!evolver_handle) return 0.0;
        return omnicode_evolver_best_fitness(evolver_handle);
    }

protected:
    static void _bind_methods() {
        ClassDB::bind_method(D_METHOD("step"), &OmnimcodeEvolverRef::step);
        ClassDB::bind_method(D_METHOD("get_generation"), &OmnimcodeEvolverRef::get_generation);
        ClassDB::bind_method(D_METHOD("get_best_fitness"), &OmnimcodeEvolverRef::get_best_fitness);
    }
};

// ============= Module Initialization =============

extern "C" {

GDExtensionBool GDE_EXPORT omnimcode_extension_init(
    const GDExtensionInterface* p_interface,
    GDExtensionClassLibraryPtr p_library,
    GDExtensionInitialization* r_initialization) {
    
    GDExtensionBinding::InitObject init_obj(p_interface, p_library, r_initialization);
    
    init_obj.register_extension_class<OmnimcodeVMRef>();
    init_obj.register_extension_class<OmnimcodeCircuitRef>();
    init_obj.register_extension_class<OmnimcodeEvolverRef>();
    
    return true;
}

GDExtensionBool GDE_EXPORT omnimcode_extension_terminate(
    const GDExtensionInterface* p_interface,
    GDExtensionClassLibraryPtr p_library,
    GDExtensionInitialization* r_initialization) {
    
    return true;
}

} // extern "C"