
import pickle
import os

def view_database():
    if not os.path.exists("face_db.pkl"):
        print("No face database found!")
        return
        
    try:
        with open("face_db.pkl", "rb") as f:
            database = pickle.load(f)
            
        print("\n=== Stored Face Data ===")
        print(f"Total people stored: {len(database)}")
        print("\nPeople in database:")
        for i, data in enumerate(database, 1):
            print(f"{i}. Name: {data['name']} (Faces stored: {len(data['encodings'])})")
            
    except Exception as e:
        print(f"Error reading database: {str(e)}")

def clear_database():
    if os.path.exists("face_db.pkl"):
        os.remove("face_db.pkl")
        print("Database cleared successfully!")
    else:
        print("No database found!")

if __name__ == "__main__":
    while True:
        print("\n1. View stored faces")
        print("2. Clear database")
        print("3. Exit")
        choice = input("Enter choice (1-3): ")
        
        if choice == "1":
            view_database()
        elif choice == "2":
            confirm = input("Are you sure you want to clear the database? (y/n): ")
            if confirm.lower() == 'y':
                clear_database()
        elif choice == "3":
            break