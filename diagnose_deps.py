import sys
import os
import subprocess
import pkg_resources

def get_user_site_packages():
    # Force looking at the user site packages (~/.local)
    # We run this as a subprocess to bypass the current env isolation if active
    cmd = [sys.executable, '-m', 'pip', 'list', '--user', '--format=json']
    try:
        import json
        result = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
        return json.loads(result)
    except Exception as e:
        print(f"Could not query user packages: {e}")
        return []

def get_env_packages():
    # Get packages in the CURRENT active Conda environment
    return {p.key: p.version for p in pkg_resources.working_set}

def main():
    print(f"--- DIAGNOSTIC REPORT ---")
    print(f"Current Environment: {sys.prefix}")
    
    user_packages = get_user_site_packages()
    env_packages = get_env_packages()
    
    missing_in_env = []
    version_mismatch = []

    print(f"\nScanning {len(user_packages)} packages in ~/.local (User Home)...")
    
    for pkg in user_packages:
        name = pkg['name']
        version = pkg['version']
        
        # Clean name for comparison
        key = name.lower().replace('-', '_')
        
        if key not in env_packages and name.lower() not in env_packages:
            missing_in_env.append(f"{name}=={version}")
        elif key in env_packages and env_packages[key] != version:
            version_mismatch.append(f"{name}: Local({version}) vs Env({env_packages[key]})")

    print(f"\n[MISSING] Found {len(missing_in_env)} packages in local that are MISSING from Conda env:")
    for p in missing_in_env:
        print(f" - {p}")

    if version_mismatch:
        print(f"\n[WARNING] Found {len(version_mismatch)} version conflicts (Safety risk):")
        for p in version_mismatch:
            print(f" - {p}")
            
    print("-" * 30)
    if missing_in_env:
        print("\n>>> SOLUTION COMMAND (Copy & Run this to fix ALL missing deps):")
        print(f"pip install --ignore-installed {' '.join(missing_in_env)}")
    else:
        print("\n>>> NO MISSING PACKAGES FOUND. Your environment is self-contained!")

if __name__ == "__main__":
    main()