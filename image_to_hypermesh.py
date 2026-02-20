def export_for_hypermesh(self, output_dir='output', prefix='common'):
        """Export common structure in multiple HyperMesh formats"""
        if self.common_structure is None:
            raise ValueError("Must call find_common_structures() first")
        
        # Create output directory
        Path(output_dir).mkdir(exist_ok=True)
        
        print("\n" + "="*60)
        print("EXPORTING FOR HYPERMESH")
        print("="*60)
        
        # 1. TCL Script for direct density assignment
        tcl_file = f"{output_dir}/{prefix}_assign_densities.tcl"
        self._export_tcl_script(tcl_file)
        
        # 2. OptiStruct include file
        inc_file = f"{output_dir}/{prefix}_initial_design.inc"
        self._export_optistruct_include(inc_file)
        
        # 3. Element sets for passive regions
        sets_file = f"{output_dir}/{prefix}_element_sets.tcl"
        self._export_element_sets(sets_file)
        
        # 4. CSV for manual import
        csv_file = f"{output_dir}/{prefix}_densities.csv"
        self._export_csv(csv_file)
        
        # 5. NumPy array
        npy_file = f"{output_dir}/{prefix}_density_field.npy"
        np.save(npy_file, self.common_structure)
        print(f"✓ Exported NumPy array: {npy_file}")
        
        print("\n" + "="*60)
        print("EXPORT COMPLETE")
        print("="*60)
        print(f"All files saved to: {output_dir}/")
    
    def _export_tcl_script(self, filename):
        """Generate TCL script for HyperMesh"""
        with open(filename, 'w') as f:
            f.write("# HyperMesh TCL Script - Assign Common Structure Densities\n")
            f.write(f"# Mesh resolution: {self.nelx} x {self.nely}\n")
            f.write(f"# Material fraction: {np.mean(self.common_structure)*100:.2f}%\n\n")
            
            f.write("*clearmark elements 1\n")
            f.write("*createmark elements 1 \"by comp name\" \"DESIGN\"\n\n")
            
            f.write("# Assign densities to elements\n")
            elem_id = 1
            for j in range(self.nely):
                for i in range(self.nelx):
                    density = self.common_structure[j, i]
                    if density > 0.001:
                        f.write(f"*attributeupdateint elements {elem_id} 3272 1 2 0 1\n")
                        f.write(f"*attributeupdatedouble elements {elem_id} 326 1 2 0 {density:.6f}\n")
                    elem_id += 1
            
            f.write("\n*clearmark elements 1\n")
        
        print(f"✓ Exported TCL script: {filename}")
    
    def _export_optistruct_include(self, filename):
        """Generate OptiStruct include file"""
        with open(filename, 'w') as f:
            f.write("$ OptiStruct Include File - Common Structure Initial Design\n")
            f.write(f"$ Mesh: {self.nelx}x{self.nely} elements\n")
            f.write(f"$ Material: {np.mean(self.common_structure)*100:.2f}%\n")
            f.write("$\n")
            
            elem_id = 1
            desvar_id = 1000
            
            for j in range(self.nely):
                for i in range(self.nelx):
                    density = self.common_structure[j, i]
                    
                    f.write(f"DESVAR  {desvar_id:8d}DV{elem_id:06d}{density:8.4f}    0.001    1.000\n")
                    f.write(f"DVPREL1 {desvar_id:8d}   PSHELL{elem_id:8d}       T\n")
                    f.write(f"        {desvar_id:8d}    1.000\n")
                    
                    elem_id += 1
                    desvar_id += 1
            
            f.write("$\n")
        
        print(f"✓ Exported OptiStruct include: {filename}")
    
    def _export_element_sets(self, filename):
        """Generate element sets TCL script"""
        solid_elem = []
        void_elem = []
        design_elem = []
        
        elem_id = 1
        for j in range(self.nely):
            for i in range(self.nelx):
                density = self.common_structure[j, i]
                
                if density > 0.9:
                    solid_elem.append(elem_id)
                elif density < 0.1:
                    void_elem.append(elem_id)
                else:
                    design_elem.append(elem_id)
                
                elem_id += 1
        
        with open(filename, 'w') as f:
            f.write("# Element sets for passive regions\n\n")
            
            if solid_elem:
                f.write("# Passive solid elements (common structure)\n")
                f.write("*createmark elements 1 ")
                f.write(" ".join(map(str, solid_elem[:500])))  # Limit for safety
                f.write("\n*createentity sets cardimage=SET_ELEM\n")
                f.write("*setvalue sets id=1 name=\"COMMON_SOLID\"\n\n")
            
            if void_elem:
                f.write("# Passive void elements\n")
                f.write("*createmark elements 1 ")
                f.write(" ".join(map(str, void_elem[:500])))
                f.write("\n*createentity sets cardimage=SET_ELEM\n")
                f.write("*setvalue sets id=2 name=\"COMMON_VOID\"\n\n")
        
        print(f"✓ Exported element sets: {filename}")
        print(f"  Solid elements: {len(solid_elem)}")
        print(f"  Void elements: {len(void_elem)}")
        print(f"  Design elements: {len(design_elem)}")
    
    def _export_csv(self, filename):
        """Export CSV file"""
        with open(filename, 'w') as f:
            f.write("ElementID,Density\n")
            elem_id = 1
            for j in range(self.nely):
                for i in range(self.nelx):
                    f.write(f"{elem_id},{self.common_structure[j, i]:.6f}\n")
                    elem_id += 1
        
        print(f"✓ Exported CSV: {filename}")