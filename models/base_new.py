# Simple parameter counting fix - copy this to replace the complex analysis in base.py

        if is_main_process():
            # Original PEFT parameter count
            self.lora_model.print_trainable_parameters()
            
            # Simple, working parameter analysis
            print("\n[PARAMETER_DEBUG] Trainable Parameter Breakdown:")
            
            try:
                # Get all trainable parameters and categorize them
                fps_mlp = []
                fps_adapter = []
                lora_regular = []
                other = []
                
                for name, param in self.named_parameters():
                    if param.requires_grad:
                        info = (name, param.shape, param.numel())
                        
                        if 'fps_conditioning' in name:
                            fps_mlp.append(info)
                        elif any(x in name for x in ['k_fps_down', 'k_fps_up', 'v_fps_down', 'v_fps_up', 'gate_alpha']):
                            fps_adapter.append(info)
                        elif 'lora' in name.lower():
                            lora_regular.append(info)
                        else:
                            other.append(info)
                
                # Calculate totals
                total_fps_mlp = sum(x[2] for x in fps_mlp)
                total_fps_adapter = sum(x[2] for x in fps_adapter) 
                total_lora = sum(x[2] for x in lora_regular)
                total_other = sum(x[2] for x in other)
                grand_total = total_fps_mlp + total_fps_adapter + total_lora + total_other
                
                print(f"  FPS MLP parameters: {len(fps_mlp):,} params = {total_fps_mlp:,} values")
                print(f"  FPS Adapter parameters: {len(fps_adapter):,} params = {total_fps_adapter:,} values")
                print(f"  Regular LoRA parameters: {len(lora_regular):,} params = {total_lora:,} values")
                print(f"  Other parameters: {len(other):,} params = {total_other:,} values")
                print(f"  TOTAL: {len(fps_mlp) + len(fps_adapter) + len(lora_regular) + len(other):,} params = {grand_total:,} values")
                
                # Show FPS details if found
                if fps_mlp:
                    print(f"\n  FPS MLP Parameter Details:")
                    for name, shape, count in fps_mlp:
                        print(f"    {name}: {list(shape)} = {count:,} params")
                
                if fps_adapter:
                    print(f"\n  FPS Adapter Parameter Details:")
                    for name, shape, count in fps_adapter:
                        print(f"    {name}: {list(shape)} = {count:,} params")
                        
            except Exception as e:
                print(f"  ERROR: {e}")
                # Basic fallback
                total = sum(p.numel() for p in self.parameters() if p.requires_grad)
                print(f"  Fallback total: {total:,} trainable parameters")