const mongoose = require('mongoose');

const userSchema = new mongoose.Schema({
  email: { 
    type: String, 
    required: true 
  },
  name: { 
    type: String, 
    required: true 
  },
  password: { 
    type: String, 
    required: true 
  },
  role: { 
    type: String, 
    default: 'DENR staff' 
  },
  isActive: { 
    type: Boolean, 
    default: true 
  },
  status: { 
    type: String, 
    enum: ['pending', 'active', 'rejected'],
    default: 'pending' 
  }
}); 

module.exports = mongoose.model('User', userSchema);