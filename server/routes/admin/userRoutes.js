const express = require('express');
const router = express.Router();
const userController = require('../../controllers/admin/userController');

//GET /get
router.get('/get', userController.getUsers);

//POST /add
router.post('/add', userController.addUser);

//PUT /id/update
router.put('/:id/update', userController.updateUserDetails);

//PUT /id/password
router.put('/:id/password', userController.updatePassword);

//PUT /id/active
router.put('/:id/active', userController.toggleActive);

//DELETE /id/delete
router.delete('/:id/delete', userController.deleteUser);

//PUT /id/approve
router.put('/:id/approve', userController.approveUser);

//DELETE /id/reject
router.delete('/:id/reject', userController.rejectUser);

module.exports = router;