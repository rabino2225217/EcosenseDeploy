const express = require('express');
const router = express.Router();
const projectController = require('../../controllers/client/projectController');

//POST /project/create
router.post('/create', projectController.createProject);

//GET /project/list
router.get('/list', projectController.listProjects);

//GET /project/:id
router.get('/:id', projectController.getProjectById);

//PUT /project/:id/rename
router.put('/:id/rename', projectController.renameProject);

//DELETE /project/:id
router.delete('/:id', projectController.deleteProject);

module.exports = router;