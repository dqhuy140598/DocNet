const express = require('express')
const app = module.exports = express()
const port = process.env.PORT || 5000
require('./db/mongoose')
const expressLayouts = require('express-ejs-layouts')
const path = require('path')
const passport = require('passport')
    // Passport Config
require('./config/passport')(passport)
    // Define paths for Express config
const publicDirectoryPath = path.join(__dirname, '../public')
const viewsPath = path.join(__dirname, './views')
const flash = require('connect-flash')
const session = require('express-session')
    //Bodyparser
app.use(express.json())
app.use(express.urlencoded({ extended: false }))
    // Setup static directory to serve
app.use(express.static(publicDirectoryPath))
    // Setup ejs engine and views location
app.set('view engine', 'ejs')
    // app.use(expressLayouts)
app.set('views', viewsPath)
    //  Express session
app.use(
    session({
        secret: 'secret',
        resave: true,
        saveUninitialized: true
    })
)

// Passport middleware
app.use(passport.initialize())
app.use(passport.session())

// Connect flash
app.use(flash())

// Global variables
app.use(function(req, res, next) {
        res.locals.success_msg = req.flash('success_msg')
        res.locals.error_msg = req.flash('error_msg')
        res.locals.error = req.flash('error')
        next()
    })
    //Routers
app.use('/', require('./routers/index'))
app.use('/ocr', require('./routers/ocr'))
app.use('/users', require('./routers/users'))

app.listen(port, () => {
    console.log(`Server is on localhost: ${port}`)
})

