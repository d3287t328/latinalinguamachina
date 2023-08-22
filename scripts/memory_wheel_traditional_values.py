class MemoryWheel:
    def __init__(self, virtues):
        self.virtues = virtues
        self.current_month = 0

    def rotate_to_month(self, month):
        self.current_month = month % len(self.virtues)

    def display(self):
        print(f"Month {self.current_month + 1}: {self.virtues[self.current_month]}")

    def get_virtue_of_month(self, month):
        return self.virtues[month % len(self.virtues)]


virtues_of_rome = [
    "Auctoritas",  # January
    "Clementia",   # February
    "Comitas",     # March
    "Constantia",  # April
    "Dignitas",    # May
    "Firmitas",    # June
    "Frugalitas",  # July
    "Gravitas",    # August
    "Honestas",    # September
    "Humanitas",   # October
    "Industria",   # November
    "Pietas",      # December
]

memory_wheel = MemoryWheel(virtues_of_rome)
memory_wheel.rotate_to_month(3)  # Rotate to April
memory_wheel.display()  # Displays: "Month 4: Constantia"
