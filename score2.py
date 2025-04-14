import os
import subprocess
import tempfile
import shutil
import requests

def evaluate_submission(submission_path):
    # Create a temporary directory for evaluation
    with tempfile.TemporaryDirectory() as temp_dir:
        # Copy the student's submission to the temporary directory
        shutil.copytree(submission_path, os.path.join(temp_dir, 'submission'))

        # Change to the submission directory
        os.chdir(os.path.join(temp_dir, 'submission'))

        # Evaluate pytest tests
        print("Running pytest...")
        try:
            result = subprocess.run(['pytest'], capture_output=True, text=True)
            print(result.stdout)
            print(result.stderr)
            if result.returncode == 0:
                print("All tests passed!")
            else:
                print("Some tests failed.")
        except Exception as e:
            print(f"Error running pytest: {e}")

        # Build the Docker image
        print("Building Docker image...")
        try:
            result = subprocess.run(['docker', 'build', '-t', 'student_app', '.'], capture_output=True, text=True)
            print(result.stdout)
            print(result.stderr)
            if result.returncode != 0:
                print("Docker build failed.")
                return
        except Exception as e:
            print(f"Error building Docker image: {e}")
            return

        # Run the Docker container
        print("Running Docker container...")
        try:
            container = subprocess.Popen(['docker', 'run', '-d', '-p', '8080:8080', 'student_app'], stdout=subprocess.PIPE)
            container_id = container.stdout.read().strip().decode('utf-8')
            print(f"Container started with ID: {container_id}")
        except Exception as e:
            print(f"Error running Docker container: {e}")
            return

        # Wait for a moment to ensure the server is up
        import time
        time.sleep(5)

        # Test the FastAPI endpoint
        print("Testing FastAPI endpoint...")
        try:
            response = requests.get('http://localhost:8080/your-endpoint')  # Replace with the actual endpoint
            print(f"Response status code: {response.status_code}")
            print(f"Response body: {response.json()}")
        except Exception as e:
            print(f"Error testing FastAPI endpoint: {e}")
        finally:
            # Stop and remove the Docker container
            subprocess.run(['docker', 'stop', container_id])
            subprocess.run(['docker', 'rm', container_id])

if __name__ == "__main__":
    # Path to the student's submission directory
    student_submission_path = 'path/to/student/submission'
    evaluate_submission(student_submission_path)

