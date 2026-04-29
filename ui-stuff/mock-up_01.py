import customtkinter


# defining classes

class MyCheckboxFrame(customtkinter.CTkFrame):
    def __init__(self, master, title, values):
        super().__init__(master)
        self.grid_columnconfigure(0, weight=1)
        self.values = values
        self.title = title
        self.checkboxes = []

        self.title = customtkinter.CTkLabel(self, text=self.title, fg_color="gray30", corner_radius=6)
        self.title.grid(row=0, column=0, padx=10, pady=(10, 0), sticky="ew")

        for i, value in enumerate(self.values):
            checkbox = customtkinter.CTkCheckBox(self, text=value)
            checkbox.grid(row=i+1, column=0, padx=10, pady=(10, 0), sticky="w")
            self.checkboxes.append(checkbox)

    def get(self):
        checked_checkboxes = []
        for checkbox in self.checkboxes:
            if checkbox.get() == 1:
                checked_checkboxes.append(checkbox.cget("text"))
        return checked_checkboxes


class MyRadiobuttonFrame(customtkinter.CTkFrame):
    def __init__(self, master, title, values):
        super().__init__(master)
        self.grid_columnconfigure(0, weight=1)
        self.values = values
        self.title = title
        self.radiobuttons = []
        self.variable = customtkinter.StringVar(value="")

        self.title = customtkinter.CTkLabel(self, text=self.title, fg_color="gray30", corner_radius=6)
        self.title.grid(row=0, column=0, padx=10, pady=(10, 0), sticky="ew")

        for i, value in enumerate(self.values):
            radiobutton = customtkinter.CTkRadioButton(self, text=value, value=value, variable=self.variable)
            radiobutton.grid(row=i + 1, column=0, padx=10, pady=(10, 0), sticky="w")
            self.radiobuttons.append(radiobutton)

    def get(self):
        return self.variable.get()

    def set(self, value):
        self.variable.set(value)

# add frames to the App

class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()

        self.title("Aiamea")
        self.geometry("640x1280")
        self.grid_columnconfigure((0, 1), weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Key entry button
        self.button = customtkinter.CTkButton(self, text="Enter OpenAI Key", command=self.key_button_callback)
        self.button.grid(row=0, column=1, padx=(0, 10), pady=(20, 10), sticky="ew", columnspan=1)

        # Scan options
        self.radiobutton_frame = MyRadiobuttonFrame(self, "PDF scan options", values=["Grab selectable text", "Scan text with OCR"])
        self.radiobutton_frame.grid(row=1, column=1, padx=(0, 10), pady=(10, 0), sticky="nsew")

        # Output Formats
        self.checkbox_frame = MyCheckboxFrame(self, "Output formats", values=["Research Information Systems (.ris)", "OpenAIRE CERIF (.xml)", "BibTeX (.bib)", "LLM writen metadata (.json)", "PDF scan output (.txt)"])
        self.checkbox_frame.grid(row=2, column=1, padx=(0, 10), pady=(10, 0), sticky="nsew")

        # COAR Type
        self.radiobutton_frame = MyRadiobuttonFrame(self, "COAR type / CRIS template", values=["Let AI determine", "Journal", "Conference contribution" ,"Book" ,"Chapter", "Report"])
        self.radiobutton_frame.grid(row=3, column=1, padx=(0, 10), pady=(10, 0), sticky="nsew")

        # Prompting exceptions
        self.checkbox_frame = MyCheckboxFrame(self, "Prompting", values=["Generate keywords", "Omit affiliations", "Omit abstract"])
        self.checkbox_frame.grid(row=4, column=1, padx=(0, 10), pady=(10, 0), sticky="nsew")

        # Initialization button
        self.button = customtkinter.CTkButton(self, text="Extract metadata", command=self.int_button_callback)
        self.button.grid(row=5, column=1, padx=(0, 10), pady=(20, 10), sticky="ew", columnspan=1)

    def int_button_callback(self):
        print("checkbox_frame:", self.checkbox_frame.get())
        print("radiobutton_frame:", self.radiobutton_frame.get())

    def key_button_callback(self):
        print("key is okay:", self.checkbox_frame.get())
        print("radiobutton_frame:", self.radiobutton_frame.get())

# run it

app = App()
app.mainloop()