import cv2
import supervision as sv
import torch
import os
import sys
from rfdetr import RFDETRBase, RFDETRLarge
from rfdetr.util.coco_classes import COCO_CLASSES
import time

def check_cuda():
    """Check CUDA availability and print detailed information."""
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA is not available. Checking environment...")
        cuda_path = os.environ.get('CUDA_PATH')
        cuda_home = os.environ.get('CUDA_HOME')
        print(f"CUDA_PATH: {cuda_path}")
        print(f"CUDA_HOME: {cuda_home}")
        print(f"PyTorch CUDA build version: {torch.version.cuda}")
        path = os.environ.get('PATH', '')
        print("Checking PATH for CUDA:")
        for p in path.split(os.pathsep): # Use os.pathsep
            if 'cuda' in p.lower():
                print(f"  Found CUDA in PATH: {p}")

def main():
    check_cuda()

    device = "cpu"
    if torch.cuda.is_available():
        try:
            device = "cuda"
            # Test if CUDA is actually working
            _ = torch.zeros(1).to(device) # Test tensor
            print(f"Successfully created tensor on {device}")

            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True # For Ampere+ GPUs
            torch.backends.cudnn.allow_tf32 = True    # For Ampere+ GPUs
            print("CUDA optimizations enabled (cudnn.benchmark, tf32)")

            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
            # os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128' # Good for fragmentation
        except Exception as e:
            print(f"Error initializing CUDA: {e}")
            print("Falling back to CPU")
            device = "cpu"
    else:
        print("CUDA not available, using CPU.")

    print(f"Using device: {device}")

    # --- MODEL CONFIGURATION ---
    # Original: 448 (14*32)
    # Try larger resolutions that are multiples of 14 (and ideally 32 if that's a patch size)
    # Option 1: 560 (14*40). 560/32 = 17.5 (not multiple of 32)
    # Option 2: 616 (14*44). 616/32 = 19.25 (not multiple of 32)
    # Option 3: 672 (14*48). 672/32 = 21 (multiple of 32!) - This seems like a good candidate if divisible by 14 and 32 matters.
    # Option 4: Or a common size like 640, and let the model internals handle it if it's flexible.
    # Let's try 640 as it's standard, or 672 if the 14 & 32 divisibility is strict.
    # Sticking to multiples of 14 as per your comment "448 is divisible by 14"
    MODEL_RESOLUTION = 560 # Try 560, 616, or even 672 if your GPU can handle it
    # MODEL_RESOLUTION = 672 # if divisible by 14 and 32 is desired (672 = 14*48 = 32*21)

    # Track which model is currently loaded (0 = base, 1 = large)
    current_model_type = 0
    model = None

    def load_model(model_type):
        """Load the specified model type (0 = base, 1 = large)"""
        nonlocal model

        # Free up memory if a model is already loaded
        if model is not None:
            del model
            if device == "cuda":
                torch.cuda.empty_cache()

        if model_type == 0:
            print(f"Loading RFDETRBase model with resolution {MODEL_RESOLUTION}...")
            if device == "cuda":
                model = RFDETRBase(
                    device=device,
                    resolution=MODEL_RESOLUTION,
                    amp=True,
                    gradient_checkpointing=False # Disable for inference speed
                )
            else:
                model = RFDETRBase(
                    device=device,
                    resolution=MODEL_RESOLUTION, # CPU might struggle more with higher res
                    amp=False
                )
        else:  # model_type == 1
            print(f"Loading RFDETRLarge model with resolution {MODEL_RESOLUTION}...")
            if device == "cuda":
                model = RFDETRLarge(
                    device=device,
                    resolution=MODEL_RESOLUTION,
                    amp=True,
                    gradient_checkpointing=False # Disable for inference speed
                )
            else:
                model = RFDETRLarge(
                    device=device,
                    resolution=MODEL_RESOLUTION, # CPU might struggle more with higher res
                    amp=False
                )
        return model

    # Load the initial model (base model)
    model = load_model(current_model_type)
    print("Model loaded successfully!")

    # --- Attempt torch.compile() for PyTorch 2.0+ ---
    compiled_model_flag = False
    if device == "cuda" and hasattr(torch, 'compile') and torch.__version__.startswith("2."):
        print("Attempting to compile model with torch.compile()...")
        try:
            # Identify the core nn.Module. This is an educated guess.
            # Check the RFDETRBase source if this doesn't work.
            module_to_compile = None
            if hasattr(model, 'model') and isinstance(model.model, torch.nn.Module):
                module_to_compile = model.model
            elif isinstance(model, torch.nn.Module): # If RFDETRBase itself is the nn.Module
                 module_to_compile = model

            if module_to_compile:
                # "reduce-overhead" is a good general choice for inference.
                # "max-autotune" might give more speedup but takes much longer to compile.
                compiled_module = torch.compile(module_to_compile, mode="reduce-overhead")

                if hasattr(model, 'model') and isinstance(model.model, torch.nn.Module):
                    model.model = compiled_module
                else: # If model itself was compiled
                    model = compiled_module
                print("Model compiled successfully with torch.compile()!")
                compiled_model_flag = True
            else:
                print("Could not find a suitable torch.nn.Module to compile in the model object. Skipping torch.compile.")
        except Exception as e:
            print(f"torch.compile() failed: {e}")

    # Verify model is on the correct device
    if device == "cuda":
        try:
            # Access a parameter to confirm it's on CUDA
            # This depends on how RFDETR stores its core PyTorch model
            core_module_for_check = model
            if compiled_model_flag and hasattr(model, '_orig_mod') : # if compiled, original module is often here
                 core_module_for_check = model._orig_mod
            elif hasattr(model, 'model') and hasattr(model.model, 'parameters'):
                core_module_for_check = model.model

            # Check if parameters exist and get the device of the first one
            params = list(core_module_for_check.parameters())
            if params:
                model_param_device = params[0].device
                print(f"Model parameters are on device: {model_param_device}")
                if str(model_param_device) != "cuda:0": # Or just check .type == 'cuda'
                    print("WARNING: Model parameters may not be on the primary CUDA device!")
            else:
                print("Model has no parameters to check, or structure is different.")

        except Exception as e:
            print(f"Could not verify model device in detail: {e}")


    print("Opening webcam...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    print("Webcam opened successfully!")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_cam = cap.get(cv2.CAP_PROP_FPS)
    print(f"Webcam resolution: {frame_width}x{frame_height}, Native FPS: {fps_cam}")

    # Check if GUI is available
    has_gui = True
    try:
        cv2.namedWindow("RF-DETR Webcam Detection", cv2.WINDOW_NORMAL)
    except cv2.error as e:
        print(f"OpenCV GUI not available: {e}")
        print("Running in headless mode (no GUI). Press Ctrl+C to stop.")
        has_gui = False

    fps_counter = 0
    fps_start_time = time.time()
    fps_display = 0.0

    # For simple profiling
    time_read, time_preprocess, time_predict, time_postprocess, time_display = 0,0,0,0,0

    print("Starting detection loop. Press 'q' to quit, 'm' to switch between Base and Large models.")
    frame_count = 0
    model_switch_requested = False

    while True:
        loop_start_time = time.time()

        # Check if model switch was requested
        if model_switch_requested:
            current_model_type = 1 - current_model_type  # Toggle between 0 and 1
            model = load_model(current_model_type)
            model_switch_requested = False
            # Reset FPS counter after model switch
            fps_counter = 0
            fps_start_time = time.time()
            fps_display = 0.0

        # 1. Read Frame
        t_start = time.perf_counter()
        success, frame = cap.read()
        if not success:
            print("Error: Failed to read frame from webcam.")
            break
        time_read = (time.perf_counter() - t_start) * 1000 # ms

        # 2. Preprocess
        t_start = time.perf_counter()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        time_preprocess = (time.perf_counter() - t_start) * 1000

        # 3. Process frame with RF-DETR
        t_start = time.perf_counter()
        with torch.no_grad():
            detections = model.predict(rgb_frame, threshold=0.45)
        time_predict = (time.perf_counter() - t_start) * 1000

        # 4. Postprocess (Annotations)
        t_start = time.perf_counter()
        labels = [
            f"{COCO_CLASSES.get(class_id, f'ID:{class_id}')} {confidence:.2f}"
            for class_id, confidence
            in zip(detections.class_id, detections.confidence)
        ]
        annotated_frame = frame.copy() # Annotate on the BGR frame
        box_annotator = sv.BoxAnnotator(thickness=2)
        label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1, text_padding=2)
        annotated_frame = box_annotator.annotate(annotated_frame, detections)
        annotated_frame = label_annotator.annotate(annotated_frame, detections, labels)
        time_postprocess = (time.perf_counter() - t_start) * 1000

        # Calculate and display FPS
        fps_counter += 1
        current_time_fps = time.time()
        if (current_time_fps - fps_start_time) >= 1.0: # Update FPS display every second
            fps_display = fps_counter / (current_time_fps - fps_start_time)
            fps_counter = 0
            fps_start_time = current_time_fps

        # Get model type info for display
        model_type_info = "Base" if current_model_type == 0 else "Large"
        device_info = "GPU" if device == "cuda" else "CPU"
        compile_info = "Compiled" if compiled_model_flag else "Not Compiled"

        # Add FPS and processing time information to the frame (if GUI is available)
        if has_gui:
            info_text_parts = [
                f"FPS: {fps_display:.1f}",
                f"Model: {model_type_info}",
                f"Res: {MODEL_RESOLUTION}px",
                f"{device_info} ({compile_info})",
            ]
            if device == "cuda":
                gpu_mem_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                gpu_mem_reserved = torch.cuda.memory_reserved() / 1024**3 # GB
                info_text_parts.append(f"VRAM(A/R): {gpu_mem_allocated:.1f}/{gpu_mem_reserved:.1f}GB")

            info_text = " | ".join(info_text_parts)
            cv2.putText(annotated_frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Display detailed timings (optional, can be noisy)
            timing_text = (f"Times(ms): Read {time_read:.1f} | Pre {time_preprocess:.1f} | "
                        f"Infer {time_predict:.1f} | Post {time_postprocess:.1f} | DispWait {time_display:.1f}")
            cv2.putText(annotated_frame, timing_text, (10, frame_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Add model switching instructions (only if GUI is available)
        if has_gui:
            instructions_text = "Press 'm' to switch models, 'q' to quit"
            cv2.putText(annotated_frame, instructions_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # 5. Display
        t_start = time.perf_counter()

        if has_gui:
            window_title = f"RF-DETR {model_type_info} Model Webcam Detection"
            cv2.setWindowTitle("RF-DETR Webcam Detection", window_title)
            cv2.imshow("RF-DETR Webcam Detection", annotated_frame)
            key = cv2.waitKey(1) & 0xFF # waitKey(1) is important for rendering
            time_display = (time.perf_counter() - t_start) * 1000 # This will mostly be the waitKey time

            if key == ord('q'):
                print("Quitting...")
                break
            elif key == ord('m'):
                print(f"Switching from {model_type_info} model to {'Large' if current_model_type == 0 else 'Base'} model...")
                model_switch_requested = True
        else:
            # In headless mode, just print detection info periodically
            time_display = (time.perf_counter() - t_start) * 1000
            if frame_count % 10 == 0:  # Print every 10 frames
                print(f"Frame {frame_count}: Detected {len(detections.class_id)} objects, FPS: {fps_display:.1f}, Model: {model_type_info}")
                # Allow model switching in headless mode via keyboard input (non-blocking)
                if hasattr(sys, 'getsizeof') and sys.stdin.isatty():  # Check if stdin is available
                    import msvcrt
                    if msvcrt.kbhit():
                        key = msvcrt.getch()
                        if key == b'm':
                            print(f"Switching from {model_type_info} model to {'Large' if current_model_type == 0 else 'Base'} model...")
                            model_switch_requested = True
                        elif key == b'q':
                            print("Quitting...")
                            break

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()
    print("Webcam test completed.")

if __name__ == "__main__":
    main()