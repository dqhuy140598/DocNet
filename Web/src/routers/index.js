const express = require('express')
const router = express.Router()
const passport = require('passport')
const bcrypt = require('bcryptjs')
const request = require('request')
var fs = require('fs');
var pdf = require('html-pdf');
var options = { format: 'letter' };

var multer = require("multer"); 

const path = require('path')

const uploadPath = path.join(path.dirname(__dirname),"../public/uploads")

const Document = require('../config/models/Document');

console.log(uploadPath)

var storage = multer.diskStorage({
    destination: function(req, file, callback){
        callback(null, uploadPath); // set the destination
    },
    filename: function(req, file, callback){
        callback(null, Date.now() + '.jpg'); // set the file name and extension
    }
});
var upload = multer({storage: storage});


const { ensureAuthenticated, forwardAuthenticated } = require('../config/models/auth')
    //User model
const User = require('../config/models/User')
router.get('/', (req, res) => res.render('home'))
router.get('/dashboard', ensureAuthenticated, (req, res) => res.render('dashboard', {
    name: req.user.name
}))

router.get('/me', ensureAuthenticated, (req, res) => res.render('user', {
    name: req.user.name
}))

router.get('/ocr', ensureAuthenticated, (req, res) => res.render('ocr', {
    name: req.user.name
}))

router.post('/ocr', upload.single('image'), (req, res) => {
    console.log(req.body.title)
    console.log(req.file)
    const imagePath = path.join(uploadPath,req.file.filename)
    request.post({uri:"http://localhost:3000/ocr",body:imagePath},function(err,respone,body){
        if(err){
            console.log(err)
            return res.send("error")
        }
        else{
            const arrayString = respone.body.split("\n")
            return res.send({"filename":req.file.filename,"result":arrayString})
        }
    })
   
})

router.get("/download/",function(req,res){
    return res.download("/home/huydao/Source/OCR-Nodejs/file.pdf","file.pdf")
})

router.post("/download",ensureAuthenticated,(req,res)=>{
        var html = req.body;
        var string = ''
        for(var i in html){
            string += i
            string += html[i]
        }
        console.log(string)
        string.replace("&nbsp;",'')
        pdf.create(string, options).toFile('/home/huydao/Source/OCR-Nodejs/file.pdf', function(err, result) {
        if (err) return console.log(err);
        console.log(result); // { filename: '/app/businesscard.pdf' }
        return res.send("ok")
      });
})

router.post("/save", async (req,res)=>{
    const data = JSON.parse(Object.keys(req.body))
    const object = {"title":data.documentTitle,"content":data.content,"imageFile":data.image,"user":req.user._id}
    const document  =  new Document(object)
    try{
        const result = await document.save()
        const user = await User.findById({_id:req.user._id})
        user.documents.push(result._id)
        const temp = await user.save()
        return res.send("ok")
    }
    catch(err){
        console.log(err)
        return res.send("error")
    }
    
})

router.get("/database",ensureAuthenticated, async (req,res)=>{
    try{
        const listDocuments = await Document.find({user:req.user._id})
        console.log(listDocuments)
        return res.render("database",{listDocuments})
    }
    catch(err){
        console.log(err);
        return res.send(err)
    }
})

router.get("/database/:id",ensureAuthenticated, async(req,res)=>{
    const docId = req.params.id;
    try{
        const document = await Document.findById(docId);
        return res.render("detail",{document});

    }
    catch(err){
        return res.send("error")
    }
})




router.get('/editor',(req,res)=>{
    return res.render('editor')
})


router.get("/keywords",(req,res)=>{
    return res.render("keywords")
})

router.post("/keywords",(req,res)=>{
    const data = JSON.parse(Object.keys(req.body))
    request.post({uri:"http://localhost:3000/keywords",body:data.text},function(err,respone,body){
        if(err){
            console.log(err)
            return res.send("error")
        }
        else{
            console.log(respone.body)
            const arrayString = respone.body.split(" ")
            return res.send({"result":arrayString})
        }
    })
})



// Resgister Handle
router.get('/login', (req, res) => res.render('login'))
router.get('/register', (req, res) => res.render('register'))

// Resgister Handle
router.post('/register', (req, res) => {
        const { name, email, username, password, password2 } = req.body
        console.log(password)
        console.log(req.body)
        var errors = []
            //Check required fields
        if (!name || !email || !password || !password2) {
            errors.push({ msg: 'Please enter all fields' });
        }
        // Check password matched
        if (password != password2) {
            errors.push({ msg: 'Passwords do not match' });
        }
        // Check password.length()
        if (password.length < 6) {
            errors.push({ msg: 'Password must be at least 6 characters' });
        }

        if (errors.length > 0) {
            res.render('register', {
                errors,
                name,
                email,
                password,
                password2
            })
        } else {
            // Validation passed
            User.findOne(({ email: email }))
                .then(user => {
                    if (user) {
                        // User exists
                        errors.push({ msg: 'Email is already registered' })
                        res.render('register', {
                            errors,
                            name,
                            email,
                            password,
                            password2
                        })
                    } else {
                        const newUser = new User({
                                name,
                                email,
                                password
                            })
                            //Hash Password
                        bcrypt.genSalt(10, (err, salt) => bcrypt.hash(newUser.password, salt, (err, hash) => {
                            if (err) throw err
                                // Set password to hashed
                            newUser.password = hash
                                // Save user
                            newUser.save()
                                .then(user => {
                                    req.flash('success_msg', 'You are now registed and can login')
                                    res.redirect('/login')
                                })
                                .catch(err => console.log(err))
                        }))
                        console.log(newUser)



                    }
                })
        }
        console.log(errors)
    })
    // Login Handle
router.post('/login', (req, res, next) => {
        passport.authenticate('local', {
            successRedirect: '/dashboard',
            failureRedirect: '/login',
            failureFlash: true
        })(req, res, next)
    })
    // Logout
router.get('/logout', (req, res) => {
    req.logout()
    req.flash('success_msg', 'You are logged out')
    res.redirect('/login')
})
module.exports = router