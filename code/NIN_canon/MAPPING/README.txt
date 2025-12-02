

AL






In this code I'm making the BR maps between arrays, channels and stimulators slightly more accessible and understandable.
The goals are:

1. Avoid future errors
2. Avoid future headaches
3. Avoid future time wasted





 The "mapping.py" script is meant to illustrate, clarify and make possible the mapping between
 Jump, channel, instance, electrode, and stimulator channel IDs.
 


   
%%%%%%%%%%%%%%%%%%%% Some info* (also in README):
   
    --- SYSTEM ---
    We have up to 1024 recording channels:
        recorded by the 16 arrays in the monkey's cortex,
        each array having a total of 64 electrodes, connected to 8 recording instances, with 4 cereplexM bank,
        with 32 channels per bank, a total of 128 channels per instance.
        The total of 32 banks (4 per instance) are connected physically through 32 Jumps,
        each two jumps having 32 jump positions connected to each array.
        When connected, the total of 16 stimulators wich A and B banks, have 32 channels per bank,
        this is, 64 channels per stimulator are connected to each array through two jumps with 32 positions per jump.
        That means that each of the stimulator channels are mapped to specific electrodes in the arrays and have 
        a corresponding recording channel within the banks in the instances
    
    
    --- JUMPS ---
    We have 32 Jumps. 
    Each jumpIndex which goes from 1 to 1024, in this order, and
    has a respective jumpNumber from 1 to 32, because it belongs to one of these jumps
    Each jumpIndex has their associated instanceChannels_128, instanceChannels_1024 channels,
    which is key to understand this mapping.
    
    
    --- INSTANCES ---
    We have 8 recording instances that in total record from 1028 channels
    Each instance has 4 banks: A, B, C and D
    Each bank has 32 channels, and in total they make 128 channels per instance.
    Bank A has channels 1 to 32
    Bank B has channels 33 to 64
    Bank C has channels 65 to 96
    Bank D has channels 97 to 128
    
    Each instance has its banks connected in the following order: DABC, which makes
    a very specific mapping between channel (recorded by instances through bankds), jump numbers, electrode
    numbers and stimulator channel numbers necessary to be clarified.
    
    Each channel has a corresponding jumpIndex number which goes from 1 to 1024
    Each channel has a corresponding jumpNumber number which goes from 1 to 32
    Each channel has a corresponding instanceChannels_128 number which goes from 1 to 128
    Each channel has a corresponding instanceChannels_1024 number  which goes from 1 to 1024
    Each channel has a corresponding instanceNumbers number which goes from 1 to 8
    Each channel has a corresponding electrodeNumbers number which goes from 1 to 64
    Each channel has a corresponding arrayNumbers number which goes from 1 to 16
    Each channel has a corresponding cereM_banks letter (D x32, A x32, Bx32 and Cx32) repeated 8 times, one per instance. 
    This order is key to understand the mapping
    Each channel has a corresponding stimBanks letter (A x32, B x32) repeated 16 times, one per stimulator. 
    This order is key to understand the mapping
    
    
    --- STIMULATORS ---
    We have up to 16 stimulators, with stimBanks A, B,
    Each Stim bank has stimChannels 1 to 32 and 33 to 64 respectively
    Each stimChannel has stimNumbers from 1 to 16 respectively

    
    --- ARRAYS ---
    We have 16 arrays. Each array has 64 electrodes.
    



    

%%%%%%%%%%%%%%%%%%%% CHATGPT 4.0 SUMMARY OF THE INFORMATION ABOVE

System Overview
Recording Channels: 
    The system supports up to 1024 recording channels across the monkey's cortex, captured by 16 arrays. 
    Each array consists of 64 electrodes.

Instances and Banks: 
    There are 8 recording instances in total, each with 4 banks (A, B, C, D), 
    contributing to 128 channels per instance. The unique order of these banks (DABC) is crucial for the mapping strategy.

Stimulators: 
    The system includes up to 16 stimulators, each with A and B banks, supporting 64 channels per stimulator. 
    These channels are intricately mapped to specific electrodes within the arrays.


Jumps: 
    The architecture utilizes 32 Jumps to physically connect the components. 
    Each jump index ranges from 1 to 1024 and corresponds to a jump number from 1 to 32. 
    These jumps are crucial for understanding the channel-to-electrode mapping.

Instances
    Configuration: 
        Each instance's 4 banks (A, B, C, D) collectively support 128 channels, 
        with a specific sequence (DABC) for connecting these banks.

    Channel Mapping: 
        Every channel within the system is associated with a specific jump index and number, 
        instance channel numbers (both within 128 and 1024 scales), instance number (1 to 8), electrode number (1 to 64), and array number (1 to 16). The order of the cereM_banks (D, A, B, C) repeated across instances and the stimBanks (A, B) repeated across stimulators are pivotal for the mapping.

Stimulators
    Bank Configuration: 
        Each stimulator's banks (A, B) have channels numbered 1 to 32 and 33 to 64, respectively. 
        These channels correspond to stim numbers from 1 to 16.
    Arrays
        Electrode Configuration: There are 16 arrays, each equipped with 64 electrodes, 
        facilitating extensive neural recording coverage.



* This information is valid only with the specific physical connections
  done for Experiment 1, and those that use the same physical connection
