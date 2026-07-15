import datetime

def generate_playful_pardon(your_name):
    offender_name = "Dhanashree"
    today_date = datetime.date.today().strftime("%B %d, %Y")
    
    # Playful Apology Message
    apology_message = f"""
    --- THE ACCUSED SPEAKS ---
    Date: {today_date}

    Hey {offender_name},

    Alright, I am officially raising the white flag. 🏳️
    
    I am writing this to formally apologize for my absolute mischief and for being your 
    unofficial, highly persistent shadow. I know that continuously following you around 
    and being a general nuisance probably pushed your patience to the absolute limit. 
    
    My bad! I promise to give your shadow a break and respect your personal space bubble 
    moving forward. To make amends for my chaotic energy, I have issued you a special 
    document below. 

    Please don't block me,
    {your_name}
    """

    # Playful Certificate of Forgiveness
    certificate = f"""
    📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜
                     OFFICIAL CERTIFICATE OF FORGIVENESS
                        (The "Stop Following Me" Edition)
    📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜
    
    This document legally and unconditionally declares that:
    
                            🌟 {offender_name} 🌟
                              
    Is hereby granted 100% total immunity and forgiveness by:
    
                            👑 {your_name} 👑
                             
    For crimes including, but not limited to:
    - Unwarranted mischief and chaotic energy.
    - Operating as a full-time, unpaid personal stalker.
    - Disrupting the peace by constantly following {your_name} around.
    
    TERMS & CONDITIONS:
    All past grudges are officially wiped clean. {offender_name} is free to roam 
    without guilt, provided she keeps a minimum distance of three steps and promises 
    to reduce the mischief by at least 15%.
    
    Signed and sealed on this day: {today_date}
    
    Chief Dispenser of Forgiveness:
    __________________________________
    {your_name}
    
    📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜
    """

    print(apology_message)
    print(certificate)

# --- Run the Code ---
# Replace 'Your Name Here' with your actual name!
generate_playful_pardon(your_name="Your Name Here")
