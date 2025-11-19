const express = require('express');
const router = express.Router();
const authController = require('../../controllers/client/authController');

//POST /register
router.post('/register', authController.registerUser);

//POST /login
router.post('/login', authController.loginUser);

//POST /logout
router.post('/logout', authController.logoutUser);

//GET /me
router.get('/me', authController.getCurrentUser);

module.exports = router;