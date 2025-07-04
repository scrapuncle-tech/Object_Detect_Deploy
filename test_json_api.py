import requests
import json

# Test script for the new JSON-based similarity detection endpoints

# Base URL for your FastAPI application
BASE_URL = "http://127.0.0.1:8001"

def test_json_from_files():
    """Test the endpoint that uses local JSON files"""
    print("Testing similarity detection with local JSON files...")
    
    url = f"{BASE_URL}/detect-similarity-from-files/"
    data = {
        "pickup_json_path": "pick.json",
        "warehouse_json_path": "ware.json", 
        "similarity_threshold": 0.75
    }
    
    try:
        response = requests.post(url, data=data)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Success!")
            print(f"Run ID: {result['run_id']}")
            print(f"Pickup Product ID: {result['pickup_product']['id']}")
            print(f"Warehouse Product ID: {result['warehouse_product']['id']}")
            print(f"Images processed - Pickup: {result['pickup_product']['images_processed']}, Warehouse: {result['warehouse_product']['images_processed']}")
            if result.get('matches'):
                print(f"Found {len(result['matches'])} matches")
            return result['run_id']
        else:
            print(f"❌ Error: {response.text}")
            return None
            
    except requests.exceptions.ConnectionError:
        print("❌ Error: Could not connect to the API. Make sure the FastAPI server is running.")
        print("Run: python app.py")
        return None
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return None

def test_json_upload():
    """Test the endpoint that accepts uploaded JSON files"""
    print("\nTesting similarity detection with uploaded JSON files...")
    
    url = f"{BASE_URL}/detect-similarity-json/"
    
    try:
        with open("pick.json", "rb") as pickup_file, open("ware.json", "rb") as warehouse_file:
            files = {
                "pickup_json": ("pick.json", pickup_file, "application/json"),
                "warehouse_json": ("ware.json", warehouse_file, "application/json")
            }
            data = {
                "similarity_threshold": 0.75
            }
            
            response = requests.post(url, files=files, data=data)
            print(f"Status Code: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print("✅ Success!")
                print(f"Run ID: {result['run_id']}")
                print(f"Pickup Product ID: {result['pickup_product']['id']}")
                print(f"Warehouse Product ID: {result['warehouse_product']['id']}")
                if result.get('matches'):
                    print(f"Found {len(result['matches'])} matches")
                return result['run_id']
            else:
                print(f"❌ Error: {response.text}")
                return None
                
    except FileNotFoundError as e:
        print(f"❌ Error: JSON file not found: {str(e)}")
        return None
    except requests.exceptions.ConnectionError:
        print("❌ Error: Could not connect to the API. Make sure the FastAPI server is running.")
        return None
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return None

def test_health_check():
    """Test the health check endpoint"""
    print("Testing API health check...")
    
    try:
        response = requests.get(f"{BASE_URL}/")
        if response.status_code == 200:
            result = response.json()
            print(f"✅ API is running: {result['message']}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Error: Could not connect to the API. Make sure the FastAPI server is running.")
        print("Run: python app.py")
        return False
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

if __name__ == "__main__":
    print("🧪 Testing Enhanced YOLO Similarity Detection API\n")
    
    # Test health check first
    if not test_health_check():
        print("\n❌ API is not running. Please start the server first with: python app.py")
        exit(1)
    
    print("\n" + "="*60)
    
    # Test local file endpoint
    run_id_1 = test_json_from_files()
    
    print("\n" + "="*60)
    
    # Test upload endpoint
    run_id_2 = test_json_upload()
    
    print("\n" + "="*60)
    print("📋 Summary:")
    if run_id_1:
        print(f"✅ Local files test passed - Run ID: {run_id_1}")
    else:
        print("❌ Local files test failed")
        
    if run_id_2:
        print(f"✅ Upload files test passed - Run ID: {run_id_2}")
    else:
        print("❌ Upload files test failed")
    
    print("\n💡 You can now use these endpoints:")
    print("1. POST /detect-similarity-from-files/ - Use local JSON files")
    print("2. POST /detect-similarity-json/ - Upload JSON files")
    print("3. Original POST /detect-similarity-upload/ - Upload image files directly")
