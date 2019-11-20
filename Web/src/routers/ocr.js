const express = require('express')
const router = express.Router()
const passport = require('passport')
const request = require('request')

const { ensureAuthenticated, forwardAuthenticated } = require('../config/models/auth')

router.post('/ocr',ensureAuthenticated,(req,res)=>{
    console.loh("in here")
    return res.send({
        "messsage":"hello world"
    })
})

module.exports = router