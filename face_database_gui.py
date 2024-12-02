
import tkinter as tk
from tkinter import ttk, messagebox
import pickle
import os

class FaceDatabaseGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition Database")
        self.root.geometry("400x500")
        
        # Create main frame
        main_frame = ttk.Frame(root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_label = ttk.Label(main_frame, text="Stored Face Database", font=('Helvetica', 16))
        title_label.grid(row=0, column=0, columnspan=2, pady=10)
        
        # Listbox to display faces
        self.face_listbox = tk.Listbox(main_frame, width=50, height=15)
        self.face_listbox.grid(row=1, column=0, columnspan=2, pady=10)
        
        # Buttons
        ttk.Button(main_frame, text="Refresh", command=self.load_database).grid(row=2, column=0, pady=5)
        ttk.Button(main_frame, text="Delete Selected", command=self.delete_selected).grid(row=2, column=1, pady=5)
        ttk.Button(main_frame, text="Clear All", command=self.clear_database).grid(row=3, column=0, columnspan=2, pady=5)
        
        # Statistics
        self.stats_label = ttk.Label(main_frame, text="", font=('Helvetica', 10))
        self.stats_label.grid(row=4, column=0, columnspan=2, pady=10)
        
        # Load initial data
        self.load_database()
    
    def load_database(self):
        self.face_listbox.delete(0, tk.END)
        try:
            if os.path.exists("face_db.pkl"):
                with open("face_db.pkl", "rb") as f:
                    database = pickle.load(f)
                
                for i, data in enumerate(database, 1):
                    self.face_listbox.insert(tk.END, 
                        f"{i}. {data['name']} ({len(data['encodings'])} faces)")
                
                self.stats_label.config(
                    text=f"Total people in database: {len(database)}")
            else:
                self.stats_label.config(text="No database found")
        except Exception as e:
            messagebox.showerror("Error", f"Error loading database: {str(e)}")
    
    def delete_selected(self):
        selection = self.face_listbox.curselection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a person to delete")
            return
            
        if messagebox.askyesno("Confirm", "Delete selected person?"):
            try:
                with open("face_db.pkl", "rb") as f:
                    database = pickle.load(f)
                
                idx = selection[0]
                del database[idx]
                
                with open("face_db.pkl", "wb") as f:
                    pickle.dump(database, f)
                
                self.load_database()
                messagebox.showinfo("Success", "Person deleted successfully")
            except Exception as e:
                messagebox.showerror("Error", f"Error deleting person: {str(e)}")
    
    def clear_database(self):
        if messagebox.askyesno("Confirm", "Clear entire database?"):
            try:
                if os.path.exists("face_db.pkl"):
                    os.remove("face_db.pkl")
                    self.load_database()
                    messagebox.showinfo("Success", "Database cleared successfully")
            except Exception as e:
                messagebox.showerror("Error", f"Error clearing database: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceDatabaseGUI(root)
    root.mainloop()